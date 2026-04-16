'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CodemapNode {
  id: string;
  title: string;
  description?: string;
  filePath?: string;
  lineStart?: number;
  lineEnd?: number;
  codeSnippet?: string;
  children?: CodemapNode[];
}

interface CodemapSection {
  id: string;
  title: string;
  description?: string;
  entryPoint?: string;
  steps: CodemapNode[];
}

interface CodemapData {
  title: string;
  summary?: string;
  sections: CodemapSection[];
  files?: string[];
}

interface CodeMapViewProps {
  codemapRaw: string;
  isLoading: boolean;
  repoUrl: string;
  repoType: string;
  repoToken?: string;
  branch?: string;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const EXT_LANG: Record<string, string> = {
  py: 'python', js: 'javascript', ts: 'typescript', tsx: 'tsx', jsx: 'jsx',
  java: 'java', go: 'go', rs: 'rust', cpp: 'cpp', c: 'c', cs: 'csharp',
  rb: 'ruby', php: 'php', swift: 'swift', kt: 'kotlin', sh: 'bash',
  bash: 'bash', yml: 'yaml', yaml: 'yaml', json: 'json', md: 'markdown',
  html: 'html', css: 'css', sql: 'sql', vue: 'jsx', scala: 'scala',
};

function getLang(fp: string) {
  return EXT_LANG[fp.split('.').pop()?.toLowerCase() ?? ''] ?? 'text';
}

function shortName(fp: string) { return fp.split('/').pop() ?? fp; }

function parseRaw(raw: string): CodemapData | null {
  const tagged = raw.match(/<CODEMAP_JSON>([\s\S]*?)<\/CODEMAP_JSON>/);
  const src = tagged ? tagged[1].trim() : raw.trim();
  const stripped = src.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
  try { return JSON.parse(stripped); } catch {}
  try {
    const m = raw.match(/\{[\s\S]*?"sections"[\s\S]*?\}(?=\s*(?:<\/CODEMAP_JSON>|$))/);
    if (m) return JSON.parse(m[0]);
  } catch {}
  return null;
}

function findById(nodes: CodemapNode[], id: string): CodemapNode | null {
  for (const n of nodes) {
    if (n.id === id) return n;
    if (n.children?.length) {
      const f = findById(n.children, id);
      if (f) return f;
    }
  }
  return null;
}

function collectFilePaths(nodes: CodemapNode[], files = new Set<string>()) {
  for (const node of nodes) {
    if (node.filePath) files.add(node.filePath);
    if (node.children?.length) collectFilePaths(node.children, files);
  }
  return files;
}

function normalizeSnippet(text?: string) {
  if (!text) return '';
  return text
    .replace(/…/g, '')
    .replace(/\.\.\.$/, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeLine(text?: string) {
  return (text ?? '').replace(/\s+/g, ' ').trim();
}

function getExtension(fp?: string) {
  return fp?.split('.').pop()?.toLowerCase() ?? '';
}

function titleHintsFunction(title?: string) {
  return /方法|函数|method|function|constructor|初始化|init|open\s*\(/i.test(title ?? '');
}

function looksLikeFunctionSignature(line: string, ext: string) {
  const trimmed = line.trim();
  if (!trimmed) return false;

  if (ext === 'py') {
    return /^(async\s+def|def)\s+[A-Za-z_][\w]*\s*\(/.test(trimmed);
  }

  if (['java', 'kt', 'scala', 'groovy'].includes(ext)) {
    return /^(public|private|protected|static|final|abstract|synchronized|native|default|override|@\w+|\w+\s+)*[A-Za-z_$][\w$<>\[\]]*\s+[A-Za-z_$][\w$]*\s*\([^;]*\)\s*(throws\s+[^{]+)?\s*\{?$/.test(trimmed);
  }

  if (['js', 'jsx', 'ts', 'tsx'].includes(ext)) {
    return /^(async\s+function|function)\s+[A-Za-z_$][\w$]*\s*\(|^[A-Za-z_$][\w$]*\s*\([^;]*\)\s*\{|^(public|private|protected|static|async|get|set)\s+[A-Za-z_$][\w$]*\s*\([^;]*\)\s*\{?$/.test(trimmed);
  }

  if (['go', 'rs', 'swift', 'php', 'cpp', 'cc', 'cxx', 'c', 'cs'].includes(ext)) {
    return /^(func\s+|fn\s+|public\s+|private\s+|protected\s+|internal\s+|static\s+|virtual\s+|override\s+|inline\s+|template\s*<|[A-Za-z_][\w:<>,*&\[\]\s]+\s+[A-Za-z_][\w]*\s*\([^;]*\))/.test(trimmed);
  }

  return false;
}

function findNextNonEmptyLine(lines: string[], startIndex: number, endIndex: number) {
  for (let index = startIndex; index <= endIndex && index < lines.length; index += 1) {
    if (lines[index].trim()) return index;
  }
  return -1;
}

function findMatchingBraceLine(lines: string[], startIndex: number) {
  let depth = 0;
  let seenOpen = false;

  for (let lineIndex = startIndex; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    for (const char of line) {
      if (char === '{') {
        depth += 1;
        seenOpen = true;
      } else if (char === '}') {
        depth -= 1;
        if (seenOpen && depth === 0) return lineIndex;
      }
    }
  }

  return startIndex;
}

function inferHighlightRange(node: CodemapNode, lines: string[], anchorIndex: number, fallbackStart: number, fallbackEnd: number) {
  const ext = getExtension(node.filePath);
  const wantsFunctionBlock = titleHintsFunction(node.title);

  if (ext === 'py') {
    for (let index = anchorIndex; index >= 0; index -= 1) {
      const line = lines[index];
      if (!line.trim()) continue;
      if (!looksLikeFunctionSignature(line, ext)) continue;

      const baseIndent = line.match(/^\s*/)?.[0].length ?? 0;
      let endIndex = anchorIndex;
      for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
        const current = lines[cursor];
        if (!current.trim()) continue;
        const indent = current.match(/^\s*/)?.[0].length ?? 0;
        if (indent <= baseIndent) break;
        endIndex = cursor;
      }

      if (index <= anchorIndex && endIndex >= anchorIndex) {
        return { highlightStart: index + 1, highlightEnd: endIndex + 1 };
      }
    }

    return { highlightStart: fallbackStart, highlightEnd: fallbackEnd };
  }

  for (let index = anchorIndex; index >= 0; index -= 1) {
    const trimmed = lines[index].trim();
    if (!trimmed) continue;

    const signatureMatch = looksLikeFunctionSignature(lines[index], ext);
    if (wantsFunctionBlock && !signatureMatch) continue;

    let braceLine = trimmed.includes('{') ? index : -1;
    if (braceLine < 0 && signatureMatch) {
      braceLine = findNextNonEmptyLine(lines, index + 1, anchorIndex);
      if (braceLine >= 0 && !lines[braceLine].includes('{')) {
        braceLine = -1;
      }
    }

    if (braceLine < 0) continue;

    const endIndex = findMatchingBraceLine(lines, braceLine);
    const highlightStart = signatureMatch ? index + 1 : braceLine + 1;
    if (highlightStart <= anchorIndex + 1 && endIndex >= anchorIndex) {
      return { highlightStart, highlightEnd: endIndex + 1 };
    }
  }

  return { highlightStart: fallbackStart, highlightEnd: fallbackEnd };
}

function resolveNodeDisplay(node: CodemapNode, fileCache: Map<string, string>) {
  if (!node.filePath) {
    return {
      lineStart: node.lineStart,
      lineEnd: node.lineEnd,
      codeSnippet: node.codeSnippet,
      highlightStart: node.lineStart,
      highlightEnd: node.lineEnd,
    };
  }

  const content = fileCache.get(node.filePath);
  if (!content) {
    return {
      lineStart: node.lineStart,
      lineEnd: node.lineEnd,
      codeSnippet: node.codeSnippet,
      highlightStart: node.lineStart,
      highlightEnd: node.lineEnd,
    };
  }

  const lines = content.split(/\r?\n/);
  const preferredIndex = node.lineStart != null ? node.lineStart - 1 : -1;
  const normalizedSnippet = normalizeSnippet(node.codeSnippet);
  const span = node.lineStart != null && node.lineEnd != null && node.lineEnd >= node.lineStart
    ? node.lineEnd - node.lineStart
    : 0;

  let resolvedIndex = preferredIndex >= 0 && preferredIndex < lines.length ? preferredIndex : -1;
  const preferredLine = preferredIndex >= 0 ? normalizeLine(lines[preferredIndex]) : '';

  if (normalizedSnippet) {
    const preferredMatches = preferredLine && (
      preferredLine.includes(normalizedSnippet) || normalizedSnippet.includes(preferredLine)
    );

    if (!preferredMatches) {
      const candidates: number[] = [];
      lines.forEach((line, index) => {
        const normalizedLine = normalizeLine(line);
        if (!normalizedLine) return;
        if (normalizedLine.includes(normalizedSnippet) || normalizedSnippet.includes(normalizedLine)) {
          candidates.push(index);
        }
      });

      if (candidates.length > 0) {
        resolvedIndex = candidates.sort((a, b) => {
          if (preferredIndex < 0) return a - b;
          return Math.abs(a - preferredIndex) - Math.abs(b - preferredIndex);
        })[0];
      }
    }
  }

  if (resolvedIndex < 0 || resolvedIndex >= lines.length) {
    return {
      lineStart: node.lineStart,
      lineEnd: node.lineEnd,
      codeSnippet: node.codeSnippet,
      highlightStart: node.lineStart,
      highlightEnd: node.lineEnd,
    };
  }

  const lineStart = resolvedIndex + 1;
  const lineEnd = Math.min(lines.length, lineStart + span);
  const codeSnippet = lines[resolvedIndex].trim() || node.codeSnippet;
  const highlightRange = inferHighlightRange(node, lines, resolvedIndex, lineStart, lineEnd);

  return {
    lineStart,
    lineEnd,
    codeSnippet,
    highlightStart: highlightRange.highlightStart,
    highlightEnd: highlightRange.highlightEnd,
  };
}

// ── Inner component (shared between normal & fullscreen) ─────────────────────

function CodeMapInner({
  data,
  selected,
  setSelected,
  expandedSections,
  setExpandedSections,
  openFiles,
  activeFile,
  setActiveFile,
  fileCache,
  fileErrors,
  fileLoading,
  scrollerRef,
  isFullscreen,
  onToggleFullscreen,
}: {
  data: CodemapData;
  selected: CodemapNode | null;
  setSelected: (n: CodemapNode) => void;
  expandedSections: Set<string>;
  setExpandedSections: (s: Set<string>) => void;
  openFiles: string[];
  activeFile: string;
  setActiveFile: (f: string) => void;
  fileCache: Map<string, string>;
  fileErrors: Map<string, string>;
  fileLoading: boolean;
  scrollerRef: React.RefObject<HTMLDivElement | null>;
  isFullscreen: boolean;
  onToggleFullscreen?: () => void;
}) {
  const resolvedSelected = useMemo(() => {
    return selected ? { ...selected, ...resolveNodeDisplay(selected, fileCache) } : null;
  }, [selected, fileCache]);

  const toggleSection = (key: string) => {
    const n = new Set(expandedSections);
    n.has(key) ? n.delete(key) : n.add(key);
    setExpandedSections(n);
  };

  // Summary with clickable refs
  const Summary = ({ text }: { text: string }) => {
    const parts = text.split(/(\[[^\]]+\])/);
    return (
      <p className="text-xs text-[var(--foreground)]/65 leading-relaxed px-4 py-2.5 border-b border-[var(--border-color)]">
        {parts.map((p, i) => {
          const m = p.match(/^\[([^\]]+)\]$/);
          if (!m) return <span key={i}>{p}</span>;
          const firstId = m[1].split('-')[0].trim();
          return (
            <button key={i} onClick={() => {
              for (const sec of data.sections) {
                const found = findById(sec.steps, firstId);
                if (found) { setSelected(found); return; }
              }
            }} className="text-purple-500 dark:text-purple-400 hover:underline font-medium">{p}</button>
          );
        })}
      </p>
    );
  };

  const Leaf = ({ node, idx }: { node: CodemapNode; idx: number }) => (
    <div key={idx} className="flex items-center gap-1.5 py-0.5 pl-1">
      <div className="w-1.5 h-1.5 rounded-full border border-gray-400 dark:border-gray-600 flex-shrink-0" />
      <span className="text-[11px] text-gray-500 dark:text-gray-400">{node.title}</span>
    </div>
  );

  const StepCard = ({ node, depth = 0 }: { node: CodemapNode; depth?: number }) => {
    const isSelected = selected?.id === node.id && selected?.title === node.title;
    const hasChildren = (node.children?.length ?? 0) > 0;
    const resolvedNode = resolveNodeDisplay(node, fileCache);
    if (!node.id && !node.filePath) return <Leaf node={node} idx={0} />;

    return (
      <div>
        <div
          onClick={() => setSelected(node)}
          className={`
            group cursor-pointer rounded-lg px-3 py-2 mb-0.5 border transition-all
            ${depth > 0 ? 'text-[0.9em]' : ''}
            ${isSelected
              ? 'bg-purple-50 dark:bg-purple-900/30 border-purple-200 dark:border-purple-700 shadow-sm'
              : 'border-transparent hover:bg-gray-50 dark:hover:bg-gray-800/40 hover:border-gray-100 dark:hover:border-gray-700/50'
            }
          `}
        >
          <div className="flex items-start gap-2">
            {node.id && (
              <span className={`flex-shrink-0 text-[10px] font-mono font-bold mt-0.5 min-w-[22px] text-center px-1 py-0.5 rounded ${
                isSelected
                  ? 'bg-purple-200 dark:bg-purple-700 text-purple-700 dark:text-purple-100'
                  : 'bg-gray-100 dark:bg-gray-700/60 text-gray-500 dark:text-gray-400'
              }`}>{node.id}</span>
            )}
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-2">
                <span className={`text-xs font-semibold leading-snug ${isSelected ? 'text-purple-700 dark:text-purple-300' : 'text-gray-800 dark:text-gray-200'}`}>
                  {node.title}
                </span>
                {node.filePath && (
                  <span className="flex-shrink-0 text-[10px] font-mono text-gray-400 dark:text-gray-500 whitespace-nowrap">
                    {shortName(node.filePath)}{resolvedNode.lineStart != null ? `:${resolvedNode.lineStart}` : ''}
                  </span>
                )}
              </div>
              {resolvedNode.codeSnippet && (
                <div className="text-[10px] font-mono text-gray-400 dark:text-gray-500 truncate mt-0.5">{resolvedNode.codeSnippet}</div>
              )}
            </div>
          </div>
        </div>
        {hasChildren && (
          <div className="ml-5 border-l-2 border-gray-100 dark:border-gray-800 pl-2 mb-1">
            {node.children!.map((child, ci) => (
              child.id || child.filePath
                ? <StepCard key={`${child.id}-${ci}`} node={child} depth={depth + 1} />
                : <Leaf key={ci} node={child} idx={ci} />
            ))}
          </div>
        )}
      </div>
    );
  };

  const currentContent = fileCache.get(activeFile) ?? '';
  const currentError = fileErrors.get(activeFile);
  const panelH = isFullscreen ? 'h-[calc(100vh-2rem)]' : 'h-[660px]';

  return (
    <div className={`flex ${panelH} rounded-lg overflow-hidden border border-[var(--border-color)]`}>

      {/* ── LEFT: Tree ─────────────────────────────────────────────────────── */}
      <div className="w-[360px] flex-shrink-0 flex flex-col border-r border-[var(--border-color)] bg-[var(--card-bg)] overflow-hidden">
        {/* Title + fullscreen toggle */}
        <div className="px-4 py-3 border-b border-[var(--border-color)] flex items-center justify-between flex-shrink-0">
          <h3 className="text-sm font-bold text-[var(--foreground)] leading-snug flex-1 mr-2">{data.title}</h3>
          {onToggleFullscreen && (
            <button
              onClick={onToggleFullscreen}
              className="flex-shrink-0 text-[var(--muted)] hover:text-[var(--foreground)] transition-colors p-1 rounded"
              title={isFullscreen ? '退出全屏' : '全屏查看'}
            >
              {isFullscreen ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
                </svg>
              )}
            </button>
          )}
        </div>

        {data.summary && <Summary text={data.summary} />}

        {/* Sections */}
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
          {data.sections.map((section, si) => {
            const key = String(si);
            const expanded = expandedSections.has(key);
            return (
              <div key={`${section.id}-${si}`} className="rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleSection(key)}
                  className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                >
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-[var(--accent-primary)] text-white text-[10px] font-bold flex items-center justify-center">
                    {section.id}
                  </span>
                  <span className="text-xs font-semibold text-[var(--foreground)] text-left flex-1 leading-snug">
                    {section.title}
                  </span>
                  <svg className={`w-3.5 h-3.5 text-[var(--muted)] transition-transform flex-shrink-0 ${expanded ? 'rotate-180' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {expanded && (
                  <div className="px-1 pb-1">
                    {section.description && (
                      <p className="text-[11px] text-[var(--muted)] px-2 pb-1 leading-relaxed">{section.description}</p>
                    )}
                    {section.entryPoint && (
                      <div className="text-[10px] font-mono text-[var(--muted)] px-2 pb-2 italic opacity-60">{section.entryPoint}</div>
                    )}
                    <div className="space-y-0.5">
                      {section.steps.map((step, si2) => (
                        <StepCard key={`${step.id}-${si2}`} node={step} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── RIGHT: Code viewer ─────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col bg-[#1a1b26] overflow-hidden min-w-0">
        {openFiles.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-4 text-gray-600">
            <svg className="w-14 h-14 opacity-15" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={0.8} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            <p className="text-sm opacity-35">点击左侧步骤查看对应代码</p>
          </div>
        ) : (
          <>
            {/* File tabs */}
            <div className="flex items-end overflow-x-auto flex-shrink-0 bg-[#16161e] border-b border-gray-700/60">
              {openFiles.map(f => (
                <button key={f} onClick={() => setActiveFile(f)}
                  className={`flex-shrink-0 px-4 py-2.5 text-[11px] font-mono transition-colors border-r border-gray-700/40 ${
                    f === activeFile
                      ? 'bg-[#1a1b26] text-white border-b-2 border-b-purple-400 -mb-px'
                      : 'text-gray-500 hover:text-gray-200 hover:bg-gray-800/60'
                  }`}
                >
                  {shortName(f)}
                  {fileErrors.has(f) && <span className="ml-1 text-red-400">!</span>}
                </button>
              ))}
            </div>

            {/* Breadcrumb */}
            {activeFile && (
              <div className="flex items-center gap-1 px-4 py-1.5 border-b border-gray-800/80 flex-shrink-0 bg-[#16161e]">
                {activeFile.split('/').map((part, i, arr) => (
                  <React.Fragment key={i}>
                    <span className={`text-[10px] font-mono ${i === arr.length - 1 ? 'text-gray-300' : 'text-gray-600'}`}>{part}</span>
                    {i < arr.length - 1 && <span className="text-[10px] text-gray-700">›</span>}
                  </React.Fragment>
                ))}
              </div>
            )}

            {/* Code */}
            <div ref={scrollerRef} className="flex-1 overflow-auto">
              {fileLoading && !currentContent && !currentError ? (
                <div className="flex items-center justify-center h-full text-gray-500 text-sm gap-2">
                  <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-gray-500 animate-spin" />
                  加载文件中…
                </div>
              ) : currentError ? (
                <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-3 px-6 text-center">
                  <svg className="w-10 h-10 opacity-30 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                  </svg>
                  <p className="text-xs text-red-400">{currentError}</p>
                  <p className="text-[11px] text-gray-600">路径：{activeFile}</p>
                </div>
              ) : currentContent ? (
                <SyntaxHighlighter
                  language={getLang(activeFile)}
                  style={tomorrow}
                  showLineNumbers
                  wrapLines
                  lineProps={(ln) => {
                    const hl = resolvedSelected?.filePath === activeFile &&
                      resolvedSelected.highlightStart != null && resolvedSelected.highlightEnd != null &&
                      ln >= resolvedSelected.highlightStart && ln <= resolvedSelected.highlightEnd;
                    const anchor = resolvedSelected?.filePath === activeFile &&
                      resolvedSelected.lineStart != null &&
                      ln === resolvedSelected.lineStart;
                    return {
                      style: {
                        display: 'block',
                        backgroundColor: anchor
                          ? 'rgba(155,124,185,0.34)'
                          : hl
                            ? 'rgba(155,124,185,0.18)'
                            : 'transparent',
                        borderLeft: anchor
                          ? '3px solid #c4a1e0'
                          : hl
                            ? '3px solid #9b7cb9'
                            : '3px solid transparent',
                      },
                    };
                  }}
                  customStyle={{ margin: 0, borderRadius: 0, fontSize: '12px', lineHeight: '19px', background: '#1a1b26', minHeight: '100%', paddingBottom: '2rem' }}
                >
                  {currentContent}
                </SyntaxHighlighter>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-700 text-sm opacity-40">无法加载文件内容</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────

export default function CodeMapView({ codemapRaw, isLoading, repoUrl, repoType, repoToken, branch, isFullscreen = false, onToggleFullscreen }: CodeMapViewProps) {
  const [data, setData] = useState<CodemapData | null>(null);
  const [selected, setSelected] = useState<CodemapNode | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['0']));
  const [openFiles, setOpenFiles] = useState<string[]>([]);
  const [activeFile, setActiveFile] = useState('');
  const [fileCache, setFileCache] = useState<Map<string, string>>(new Map());
  const [fileErrors, setFileErrors] = useState<Map<string, string>>(new Map());
  const [fileLoading, setFileLoading] = useState(false);
  const scrollerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (data) return;
    const parsed = parseRaw(codemapRaw);
    if (parsed?.sections?.length) {
      setData(parsed);
      setExpandedSections(new Set(['0']));
    }
  }, [codemapRaw, data]);

  const fetchFileContent = useCallback(async (fp: string, showLoading = false) => {
    if (!fp || fileCache.has(fp) || fileErrors.has(fp)) return;

    if (showLoading) setFileLoading(true);
    try {
      const p = new URLSearchParams({ repo_url: repoUrl, file_path: fp, type: repoType });
      if (repoToken) p.set('token', repoToken);
      if (branch) p.set('branch', branch);
      const res = await fetch(`/api/file-content?${p}`);
      if (res.ok) {
        const json = await res.json();
        setFileCache(prev => prev.has(fp) ? prev : new Map(prev).set(fp, json.content ?? ''));
      } else {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        setFileErrors(prev => prev.has(fp) ? prev : new Map(prev).set(fp, err.detail ?? `HTTP ${res.status}`));
      }
    } catch (e) {
      setFileErrors(prev => prev.has(fp) ? prev : new Map(prev).set(fp, String(e)));
    } finally {
      if (showLoading) setFileLoading(false);
    }
  }, [repoUrl, repoType, repoToken, branch, fileCache, fileErrors]);

  useEffect(() => {
    if (!data) return;
    const paths = new Set<string>();
    data.sections.forEach(section => collectFilePaths(section.steps, paths));
    paths.forEach(fp => {
      void fetchFileContent(fp, false);
    });
  }, [data, fetchFileContent]);

  const openFile = useCallback(async (fp: string) => {
    if (!fp) return;
    setActiveFile(fp);
    setOpenFiles(prev => prev.includes(fp) ? prev : [...prev, fp].slice(-8));
    await fetchFileContent(fp, true);
  }, [fetchFileContent]);

  const resolvedSelected = useMemo(() => {
    return selected ? { ...selected, ...resolveNodeDisplay(selected, fileCache) } : null;
  }, [selected, fileCache]);

  useEffect(() => {
    if (selected?.filePath) openFile(selected.filePath);
  }, [selected]); // eslint-disable-line

  useEffect(() => {
    if (!resolvedSelected || resolvedSelected.lineStart == null || !scrollerRef.current) return;
    const lineStart = resolvedSelected.lineStart;
    const timer = setTimeout(() => {
      if (scrollerRef.current) scrollerRef.current.scrollTop = Math.max(0, (lineStart - 5) * 19);
    }, 80);
    return () => clearTimeout(timer);
  }, [resolvedSelected, activeFile]);

  const handleSelect = (node: CodemapNode) => setSelected(node);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-28 text-sm text-[var(--muted)] gap-2">
        {(isLoading || codemapRaw) && <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-teal-500 animate-spin" />}
        <span>{isLoading ? '正在生成代码调用链路…' : codemapRaw ? '解析中…' : ''}</span>
      </div>
    );
  }

  const inner = (
    <CodeMapInner
      data={data} selected={selected} setSelected={handleSelect}
      expandedSections={expandedSections} setExpandedSections={setExpandedSections}
      openFiles={openFiles} activeFile={activeFile} setActiveFile={setActiveFile}
      fileCache={fileCache} fileErrors={fileErrors} fileLoading={fileLoading}
      scrollerRef={scrollerRef} isFullscreen={isFullscreen} onToggleFullscreen={onToggleFullscreen}
    />
  );

  if (isFullscreen) {
    return (
      <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
        <div className="w-full h-full max-w-[1600px]">{inner}</div>
      </div>
    );
  }

  return inner;
}
