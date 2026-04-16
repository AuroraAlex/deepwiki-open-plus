'use client';

import React, {useState, useRef, useEffect, useCallback} from 'react';
import {FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import Markdown from './Markdown';
import CodeMapView from './CodeMapView';
import { useLanguage } from '@/contexts/LanguageContext';
import RepoInfo from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import ModelSelectionModal from './ModelSelectionModal';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest } from '@/utils/websocketClient';

interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ResearchStage {
  title: string;
  content: string;
  iteration: number;
  type: 'plan' | 'update' | 'conclusion';
}

interface SavedConversation {
  id: string;
  title: string;
  timestamp: number;
  question: string;
  response: string;
  conversationHistory: Message[];
  repoUrl: string;
  codeMapContent?: string;
}

interface AskProps {
  repoInfo: RepoInfo;
  provider?: string;
  model?: string;
  isCustomModel?: boolean;
  customModel?: string;
  language?: string;
  branch?: string;
  onRef?: (ref: { clearConversation: () => void }) => void;
}

const HISTORY_KEY = 'deepwiki_conversation_history';
const MAX_HISTORY = 50;

function loadHistory(): SavedConversation[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveToHistory(conv: SavedConversation) {
  try {
    const history = loadHistory();
    const updated = [conv, ...history.filter(c => c.id !== conv.id)].slice(0, MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  } catch {
    // ignore storage errors
  }
}

function deleteFromHistory(id: string) {
  try {
    const history = loadHistory().filter(c => c.id !== id);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // ignore
  }
}

const Ask: React.FC<AskProps> = ({
  repoInfo,
  provider = '',
  model = '',
  isCustomModel = false,
  customModel = '',
  language = 'en',
  branch,
  onRef
}) => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);

  // Codemap state
  const [codeMap, setCodeMap] = useState(false);
  const [codeMapContent, setCodeMapContent] = useState('');
  const [codeMapLoading, setCodeMapLoading] = useState(false);
  const [codeMapExpanded, setCodeMapExpanded] = useState(true);
  const [codeMapFullscreen, setCodeMapFullscreen] = useState(false);
  const currentConvIdRef = useRef<string>('');

  // History panel state
  const [historyOpen, setHistoryOpen] = useState(false);
  const [savedHistory, setSavedHistory] = useState<SavedConversation[]>([]);

  // Model selection state
  const [selectedProvider, setSelectedProvider] = useState(provider);
  const [selectedModel, setSelectedModel] = useState(model);
  const [isCustomSelectedModel, setIsCustomSelectedModel] = useState(isCustomModel);
  const [customSelectedModel, setCustomSelectedModel] = useState(customModel);
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);
  const [isComprehensiveView, setIsComprehensiveView] = useState(true);

  const { messages } = useLanguage();

  // Research navigation state
  const [researchStages, setResearchStages] = useState<ResearchStage[]>([]);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [researchIteration, setResearchIteration] = useState(0);
  const [researchComplete, setResearchComplete] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const responseRef = useRef<HTMLDivElement>(null);
  const providerRef = useRef(provider);
  const modelRef = useRef(model);

  useEffect(() => {
    if (inputRef.current) inputRef.current.focus();
  }, []);

  useEffect(() => {
    if (onRef) onRef({ clearConversation });
  }, [onRef]);

  useEffect(() => {
    if (responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [response]);

  useEffect(() => {
    return () => { closeWebSocket(webSocketRef.current); };
  }, []);

  useEffect(() => {
    providerRef.current = provider;
    modelRef.current = model;
  }, [provider, model]);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        setIsLoading(true);
        const res = await fetch('/api/models/config');
        if (!res.ok) throw new Error(`Error fetching model configurations: ${res.status}`);
        const data = await res.json();
        if (providerRef.current === '' || modelRef.current === '') {
          setSelectedProvider(data.defaultProvider);
          const sp = data.providers.find((p: Provider) => p.id === data.defaultProvider);
          if (sp && sp.models.length > 0) setSelectedModel(sp.models[0].id);
        } else {
          setSelectedProvider(providerRef.current);
          setSelectedModel(modelRef.current);
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
      } finally {
        setIsLoading(false);
      }
    };
    if (provider === '' || model === '') fetchModel();
  }, [provider, model]);

  // ── Codemap helper ────────────────────────────────────────────────────────
  const fetchCodeMap = useCallback(async (
    question: string,
    history: Message[],
    prov: string,
    mdl: string
  ) => {
    setCodeMapContent('');
    setCodeMapLoading(true);
    setCodeMapExpanded(true);
    try {
      const requestBody = {
        repo_url: getRepoUrl(repoInfo),
        type: repoInfo.type,
        messages: [...history, { role: 'user', content: question }],
        provider: prov,
        model: isCustomSelectedModel ? customSelectedModel : mdl,
        language,
        ...(repoInfo?.token ? { token: repoInfo.token } : {}),
        ...(branch ? { branch } : {}),
      };
      const res = await fetch('/api/codemap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      if (!res.ok || !res.body) {
        setCodeMapContent('> Failed to generate codemap.');
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        full += decoder.decode(value, { stream: true });
        setCodeMapContent(full);
      }
    } catch (e) {
      setCodeMapContent(`> Error: ${e}`);
    } finally {
      setCodeMapLoading(false);
    }
  }, [repoInfo, isCustomSelectedModel, customSelectedModel, language]);

  // ── History helpers ───────────────────────────────────────────────────────
  const openHistory = () => {
    setSavedHistory(loadHistory());
    setHistoryOpen(true);
  };

  const restoreConversation = (conv: SavedConversation) => {
    setQuestion(conv.question);
    setResponse(conv.response);
    setConversationHistory(conv.conversationHistory);
    setResearchIteration(0);
    setResearchComplete(true);
    setResearchStages([]);
    setCurrentStageIndex(0);
    setCodeMapContent(conv.codeMapContent || '');
    setCodeMapExpanded(true);
    setHistoryOpen(false);
  };

  const clearConversation = () => {
    setQuestion('');
    setResponse('');
    setConversationHistory([]);
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setCurrentStageIndex(0);
    setCodeMapContent('');
    setCodeMapFullscreen(false);
    if (inputRef.current) inputRef.current.focus();
  };

  const downloadresponse = () => {
    const blob = new Blob([response], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `response-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // ── Research helpers ──────────────────────────────────────────────────────
  const checkIfResearchComplete = (content: string): boolean => {
    if (content.includes('## Final Conclusion')) return true;
    if ((content.includes('## Conclusion') || content.includes('## Summary')) &&
      !content.includes('I will now proceed to') &&
      !content.includes('Next Steps') &&
      !content.includes('next iteration')) return true;
    if (content.includes('This concludes our research') ||
      content.includes('This completes our investigation') ||
      content.includes('This concludes the deep research process') ||
      content.includes('Key Findings and Implementation Details') ||
      content.includes('In conclusion,') ||
      (content.includes('Final') && content.includes('Conclusion'))) return true;
    if (content.includes('Dockerfile') &&
      (content.includes('This Dockerfile') || content.includes('The Dockerfile')) &&
      !content.includes('Next Steps') &&
      !content.includes('In the next iteration')) return true;
    return false;
  };

  const extractResearchStage = (content: string, iteration: number): ResearchStage | null => {
    if (iteration === 1 && content.includes('## Research Plan')) {
      const planMatch = content.match(/## Research Plan([\s\S]*?)(?:## Next Steps|$)/);
      if (planMatch) return { title: 'Research Plan', content, iteration: 1, type: 'plan' };
    }
    if (iteration >= 1 && iteration <= 4) {
      const updateMatch = content.match(new RegExp(`## Research Update ${iteration}([\\s\\S]*?)(?:## Next Steps|$)`));
      if (updateMatch) return { title: `Research Update ${iteration}`, content, iteration, type: 'update' };
    }
    if (content.includes('## Final Conclusion')) {
      const conclusionMatch = content.match(/## Final Conclusion([\s\S]*?)$/);
      if (conclusionMatch) return { title: 'Final Conclusion', content, iteration, type: 'conclusion' };
    }
    return null;
  };

  const navigateToStage = (index: number) => {
    if (index >= 0 && index < researchStages.length) {
      setCurrentStageIndex(index);
      setResponse(researchStages[index].content);
    }
  };
  const navigateToNextStage = () => { if (currentStageIndex < researchStages.length - 1) navigateToStage(currentStageIndex + 1); };
  const navigateToPreviousStage = () => { if (currentStageIndex > 0) navigateToStage(currentStageIndex - 1); };

  const webSocketRef = useRef<WebSocket | null>(null);

  const continueResearch = async () => {
    if (!deepResearch || researchComplete || !response || isLoading) return;
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsLoading(true);
    try {
      const currentResponse = response;
      const newHistory: Message[] = [
        ...conversationHistory,
        { role: 'assistant', content: currentResponse },
        { role: 'user', content: '[DEEP RESEARCH] Continue the research' }
      ];
      setConversationHistory(newHistory);
      const newIteration = researchIteration + 1;
      setResearchIteration(newIteration);
      setResponse('');

      const requestBody: ChatCompletionRequest = {
        repo_url: getRepoUrl(repoInfo),
        type: repoInfo.type,
        messages: newHistory.map(msg => ({ role: msg.role as 'user' | 'assistant', content: msg.content })),
        provider: selectedProvider,
        model: isCustomSelectedModel ? customSelectedModel : selectedModel,
        language,
        ...(repoInfo?.token ? { token: repoInfo.token } : {}),
        ...(branch ? { branch } : {}),
      };

      closeWebSocket(webSocketRef.current);
      let fullResponse = '';

      webSocketRef.current = createChatWebSocket(
        requestBody,
        (message: string) => {
          fullResponse += message;
          setResponse(fullResponse);
          if (deepResearch) {
            const stage = extractResearchStage(fullResponse, newIteration);
            if (stage) {
              setResearchStages(prev => {
                const idx = prev.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
                if (idx >= 0) { const n = [...prev]; n[idx] = stage; return n; }
                return [...prev, stage];
              });
              setCurrentStageIndex(researchStages.length);
            }
          }
        },
        (error: Event) => {
          console.error('WebSocket error:', error);
          setResponse(prev => prev + '\n\nError: WebSocket connection failed. Falling back to HTTP...');
          fallbackToHttp(requestBody);
        },
        () => {
          const isComplete = checkIfResearchComplete(fullResponse);
          const forceComplete = newIteration >= 5;
          if (forceComplete && !isComplete) {
            const note = "\n\n## Final Conclusion\nAfter multiple iterations of deep research, we've gathered significant insights about this topic. This concludes our investigation process, having reached the maximum number of research iterations. The findings presented across all iterations collectively form our comprehensive answer to the original question.";
            fullResponse += note;
            setResponse(fullResponse);
            setResearchComplete(true);
          } else {
            setResearchComplete(isComplete);
          }
          setIsLoading(false);
        }
      );
    } catch (error) {
      console.error('Error during API call:', error);
      setResponse(prev => prev + '\n\nError: Failed to continue research. Please try again.');
      setResearchComplete(true);
      setIsLoading(false);
    }
  };

  const fallbackToHttp = async (requestBody: ChatCompletionRequest) => {
    try {
      const apiResponse = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      if (!apiResponse.ok) throw new Error(`API error: ${apiResponse.status}`);
      const reader = apiResponse.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error('Failed to get response reader');
      let fullResponse = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        fullResponse += decoder.decode(value, { stream: true });
        setResponse(fullResponse);
        if (deepResearch) {
          const stage = extractResearchStage(fullResponse, researchIteration);
          if (stage) {
            setResearchStages(prev => {
              const idx = prev.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
              if (idx >= 0) { const n = [...prev]; n[idx] = stage; return n; }
              return [...prev, stage];
            });
          }
        }
      }
      const isComplete = checkIfResearchComplete(fullResponse);
      const forceComplete = researchIteration >= 5;
      if (forceComplete && !isComplete) {
        const note = "\n\n## Final Conclusion\nAfter multiple iterations of deep research, we've gathered significant insights about this topic. This concludes our investigation process, having reached the maximum number of research iterations. The findings presented across all iterations collectively form our comprehensive answer to the original question.";
        fullResponse += note;
        setResponse(fullResponse);
        setResearchComplete(true);
      } else {
        setResearchComplete(isComplete);
      }
    } catch (error) {
      console.error('Error during HTTP fallback:', error);
      setResponse(prev => prev + '\n\nError: Failed to get a response. Please try again.');
      setResearchComplete(true);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (deepResearch && response && !isLoading && !researchComplete) {
      const isComplete = checkIfResearchComplete(response);
      if (isComplete) {
        setResearchComplete(true);
      } else if (researchIteration > 0 && researchIteration < 5) {
        const timer = setTimeout(() => { continueResearch(); }, 1000);
        return () => clearTimeout(timer);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [response, isLoading, deepResearch, researchComplete, researchIteration]);

  useEffect(() => {
    if (deepResearch && response && !isLoading) {
      const stage = extractResearchStage(response, researchIteration);
      if (stage) {
        setResearchStages(prev => {
          const idx = prev.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
          if (idx >= 0) { const n = [...prev]; n[idx] = stage; return n; }
          return [...prev, stage];
        });
        setCurrentStageIndex(prev => {
          const newIndex = researchStages.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
          return newIndex >= 0 ? newIndex : prev;
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [response, isLoading, deepResearch, researchIteration]);

  // Save conversation to history when response completes
  useEffect(() => {
    if (response && !isLoading && question) {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      currentConvIdRef.current = id;
      const conv: SavedConversation = {
        id,
        title: question.slice(0, 60) + (question.length > 60 ? '…' : ''),
        timestamp: Date.now(),
        question,
        response,
        conversationHistory,
        repoUrl: getRepoUrl(repoInfo),
        codeMapContent: codeMapContent || undefined,
      };
      saveToHistory(conv);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading]);

  // Update history entry with codemap content once codemap finishes loading
  useEffect(() => {
    if (!codeMapLoading && codeMapContent && currentConvIdRef.current) {
      const history = loadHistory();
      const existing = history.find(c => c.id === currentConvIdRef.current);
      if (existing) {
        saveToHistory({ ...existing, codeMapContent });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [codeMapLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    handleConfirmAsk();
  };

  const handleConfirmAsk = async () => {
    setIsLoading(true);
    setResponse('');
    setCodeMapContent('');
    setResearchIteration(0);
    setResearchComplete(false);

    try {
      const initialMessage: Message = {
        role: 'user',
        content: deepResearch ? `[DEEP RESEARCH] ${question}` : question,
      };
      const newHistory: Message[] = [initialMessage];
      setConversationHistory(newHistory);

      const requestBody: ChatCompletionRequest = {
        repo_url: getRepoUrl(repoInfo),
        type: repoInfo.type,
        messages: newHistory.map(msg => ({ role: msg.role as 'user' | 'assistant', content: msg.content })),
        provider: selectedProvider,
        model: isCustomSelectedModel ? customSelectedModel : selectedModel,
        language,
        ...(repoInfo?.token ? { token: repoInfo.token } : {}),
        ...(branch ? { branch } : {}),
      };

      closeWebSocket(webSocketRef.current);
      let fullResponse = '';

      webSocketRef.current = createChatWebSocket(
        requestBody,
        (message: string) => {
          fullResponse += message;
          setResponse(fullResponse);
          if (deepResearch) {
            const stage = extractResearchStage(fullResponse, 1);
            if (stage) { setResearchStages([stage]); setCurrentStageIndex(0); }
          }
        },
        (error: Event) => {
          console.error('WebSocket error:', error);
          setResponse(prev => prev + '\n\nError: WebSocket connection failed. Falling back to HTTP...');
          fallbackToHttp(requestBody);
        },
        () => {
          if (deepResearch) {
            const isComplete = checkIfResearchComplete(fullResponse);
            setResearchComplete(isComplete);
            if (!isComplete) setResearchIteration(1);
          }
          setIsLoading(false);
          // Trigger codemap after response completes
          if (codeMap) {
            fetchCodeMap(question, [], selectedProvider, isCustomSelectedModel ? customSelectedModel : selectedModel);
          }
        }
      );
    } catch (error) {
      console.error('Error during API call:', error);
      setResponse(prev => prev + '\n\nError: Failed to get a response. Please try again.');
      setResearchComplete(true);
      setIsLoading(false);
    }
  };

  const [buttonWidth, setButtonWidth] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (buttonRef.current) setButtonWidth(buttonRef.current.offsetWidth);
  }, [messages.ask?.askButton, isLoading]);

  return (
    <div>
      <div className="p-4">
        {/* Top toolbar */}
        <div className="flex items-center justify-between mb-4">
          {/* History button */}
          <button
            type="button"
            onClick={openHistory}
            className="text-xs px-2.5 py-1 rounded border border-[var(--border-color)]/40 bg-[var(--background)]/10 text-[var(--foreground)]/80 hover:bg-[var(--background)]/30 hover:text-[var(--foreground)] transition-colors flex items-center gap-1.5"
            title="历史对话"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>历史</span>
          </button>

          {/* Model selection button */}
          <button
            type="button"
            onClick={() => setIsModelSelectionModalOpen(true)}
            className="text-xs px-2.5 py-1 rounded border border-[var(--border-color)]/40 bg-[var(--background)]/10 text-[var(--foreground)]/80 hover:bg-[var(--background)]/30 hover:text-[var(--foreground)] transition-colors flex items-center gap-1.5"
          >
            <span>{selectedProvider}/{isCustomSelectedModel ? customSelectedModel : selectedModel}</span>
            <svg className="h-3.5 w-3.5 text-[var(--accent-primary)]/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </button>
        </div>

        {/* Question input */}
        <form onSubmit={handleSubmit} className="mt-4">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={messages.ask?.placeholder || 'What would you like to know about this codebase?'}
              className="block w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] text-[var(--foreground)] px-5 py-3.5 text-base shadow-sm focus:border-[var(--accent-primary)] focus:ring-2 focus:ring-[var(--accent-primary)]/30 focus:outline-none transition-all"
              style={{ paddingRight: `${buttonWidth + 24}px` }}
              disabled={isLoading}
            />
            <button
              ref={buttonRef}
              type="submit"
              disabled={isLoading || !question.trim()}
              className={`absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-2 rounded-md font-medium text-sm ${
                isLoading || !question.trim()
                  ? 'bg-[var(--button-disabled-bg)] text-[var(--button-disabled-text)] cursor-not-allowed'
                  : 'bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/90 shadow-sm'
              } transition-all duration-200 flex items-center gap-1.5`}
            >
              {isLoading ? (
                <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-white animate-spin" />
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                  <span>{messages.ask?.askButton || 'Ask'}</span>
                </>
              )}
            </button>
          </div>

          {/* Toggles row */}
          <div className="flex items-center mt-2 gap-4 flex-wrap">
            {/* Deep Research toggle */}
            <div className="group relative">
              <label className="flex items-center cursor-pointer">
                <span className="text-xs text-gray-600 dark:text-gray-400 mr-2">Deep Research</span>
                <div className="relative">
                  <input type="checkbox" checked={deepResearch} onChange={() => setDeepResearch(!deepResearch)} className="sr-only" />
                  <div className={`w-10 h-5 rounded-full transition-colors ${deepResearch ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}`}></div>
                  <div className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform transform ${deepResearch ? 'translate-x-5' : ''}`}></div>
                </div>
              </label>
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-72 z-10">
                <div className="relative">
                  <div className="absolute -bottom-2 left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
                  <p className="mb-1">Deep Research conducts a multi-turn investigation process.</p>
                </div>
              </div>
            </div>

            {/* Codemap toggle */}
            <div className="group relative">
              <label className="flex items-center cursor-pointer">
                <span className="text-xs text-gray-600 dark:text-gray-400 mr-2">代码调用链</span>
                <div className="relative">
                  <input type="checkbox" checked={codeMap} onChange={() => setCodeMap(!codeMap)} className="sr-only" />
                  <div className={`w-10 h-5 rounded-full transition-colors ${codeMap ? 'bg-teal-600' : 'bg-gray-300 dark:bg-gray-600'}`}></div>
                  <div className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform transform ${codeMap ? 'translate-x-5' : ''}`}></div>
                </div>
              </label>
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-64 z-10">
                <div className="relative">
                  <div className="absolute -bottom-2 left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
                  <p>开启后，在 AI 回答的同时生成代码调用链路 Mermaid 图，直观展示函数/模块间的依赖关系。</p>
                </div>
              </div>
            </div>

            {/* Status labels */}
            <div className="ml-auto flex items-center gap-2">
              {deepResearch && (
                <span className="text-xs text-purple-600 dark:text-purple-400">
                  Multi-turn research
                  {researchIteration > 0 && !researchComplete && ` (iter ${researchIteration})`}
                  {researchComplete && ` (complete)`}
                </span>
              )}
              {codeMap && (
                <span className="text-xs text-teal-600 dark:text-teal-400 flex items-center gap-1">
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                  Codemap ON
                </span>
              )}
            </div>
          </div>
        </form>

        {/* Response area */}
        {response && (
          <div className="border-t border-gray-200 dark:border-gray-700 mt-4">
            <div ref={responseRef} className="p-4 max-h-[500px] overflow-y-auto">
              <Markdown content={response} />
            </div>

            {/* Bottom toolbar */}
            <div className="p-2 flex justify-between items-center border-t border-gray-200 dark:border-gray-700">
              {/* Research navigation */}
              {deepResearch && researchStages.length > 1 && (
                <div className="flex items-center space-x-2">
                  <button onClick={navigateToPreviousStage} disabled={currentStageIndex === 0}
                    className={`p-1 rounded-md ${currentStageIndex === 0 ? 'text-gray-400 dark:text-gray-600' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'}`}>
                    <FaChevronLeft size={12} />
                  </button>
                  <div className="text-xs text-gray-600 dark:text-gray-400">{currentStageIndex + 1} / {researchStages.length}</div>
                  <button onClick={navigateToNextStage} disabled={currentStageIndex === researchStages.length - 1}
                    className={`p-1 rounded-md ${currentStageIndex === researchStages.length - 1 ? 'text-gray-400 dark:text-gray-600' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'}`}>
                    <FaChevronRight size={12} />
                  </button>
                  <div className="text-xs text-gray-600 dark:text-gray-400 ml-2">
                    {researchStages[currentStageIndex]?.title || `Stage ${currentStageIndex + 1}`}
                  </div>
                </div>
              )}

              <div className="flex items-center space-x-2 ml-auto">
                {/* Download button */}
                <button onClick={downloadresponse}
                  className="text-xs text-gray-500 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-1"
                  title="下载为 Markdown">
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Download
                </button>
                {/* Clear button */}
                <button id="ask-clear-conversation" onClick={clearConversation}
                  className="text-xs text-gray-500 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700">
                  Clear conversation
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Codemap panel */}
        {(codeMapLoading || codeMapContent) && (
          <div className="mt-4">
            {/* Panel header */}
            <div className="flex items-center justify-between px-4 py-2 mb-2 bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 rounded-lg">
              <button
                onClick={() => setCodeMapExpanded(e => !e)}
                className="flex items-center gap-2 text-teal-800 dark:text-teal-200 text-sm font-medium flex-1 text-left"
              >
                <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
                <span>代码调用链路</span>
                {codeMapLoading && (
                  <div className="w-3 h-3 rounded-full border-2 border-t-transparent border-teal-500 animate-spin ml-1" />
                )}
                <svg className={`w-4 h-4 transition-transform ml-1 ${codeMapExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {/* Fullscreen button */}
              <button
                onClick={() => { setCodeMapExpanded(true); setCodeMapFullscreen(f => !f); }}
                className="ml-2 p-1 text-teal-600 dark:text-teal-400 hover:text-teal-800 dark:hover:text-teal-200 transition-colors"
                title="全屏查看"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
                </svg>
              </button>
            </div>

            {/* Interactive codemap */}
            {codeMapExpanded && !codeMapFullscreen && (
              <CodeMapView
                codemapRaw={codeMapContent}
                isLoading={codeMapLoading}
                repoUrl={getRepoUrl(repoInfo)}
                repoType={repoInfo.type || 'github'}
                repoToken={repoInfo.token ?? undefined}
                branch={branch}
                isFullscreen={false}
                onToggleFullscreen={() => setCodeMapFullscreen(true)}
              />
            )}
          </div>
        )}

        {/* Fullscreen codemap overlay */}
        {codeMapFullscreen && (
          <CodeMapView
            codemapRaw={codeMapContent}
            isLoading={codeMapLoading}
            repoUrl={getRepoUrl(repoInfo)}
            repoType={repoInfo.type || 'github'}
            repoToken={repoInfo.token ?? undefined}
            branch={branch}
            isFullscreen={true}
            onToggleFullscreen={() => setCodeMapFullscreen(false)}
          />
        )}

        {/* Loading indicator */}
        {isLoading && !response && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <div className="animate-pulse flex space-x-1">
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {deepResearch
                  ? (researchIteration === 0 ? "Planning research approach..." : `Research iteration ${researchIteration} in progress...`)
                  : "Thinking..."}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* ── History Drawer ──────────────────────────────────────────────── */}
      {historyOpen && (
        <div className="fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/40" onClick={() => setHistoryOpen(false)} />
          {/* Panel */}
          <div className="relative ml-auto w-80 max-w-full h-full bg-[var(--card-bg)] shadow-xl flex flex-col border-l border-[var(--border-color)]">
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-color)]">
              <h2 className="text-sm font-semibold text-[var(--foreground)]">历史对话</h2>
              <button onClick={() => setHistoryOpen(false)} className="text-[var(--muted)] hover:text-[var(--foreground)]">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              {savedHistory.length === 0 ? (
                <div className="p-6 text-center text-sm text-[var(--muted)]">暂无历史记录</div>
              ) : (
                savedHistory.map(conv => (
                  <div key={conv.id} className="group border-b border-[var(--border-color)] px-4 py-3 hover:bg-[var(--background)]/50 transition-colors">
                    <div className="flex items-start justify-between gap-2">
                      <button className="flex-1 text-left" onClick={() => restoreConversation(conv)}>
                        <div className="text-xs font-medium text-[var(--foreground)] line-clamp-2 leading-relaxed">{conv.title}</div>
                        <div className="text-[10px] text-[var(--muted)] mt-1">
                          {new Date(conv.timestamp).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                          {' · '}
                          <span className="opacity-70">{conv.repoUrl.split('/').slice(-2).join('/')}</span>
                        </div>
                      </button>
                      <button
                        onClick={() => {
                          deleteFromHistory(conv.id);
                          setSavedHistory(prev => prev.filter(c => c.id !== conv.id));
                        }}
                        className="opacity-0 group-hover:opacity-100 text-[var(--muted)] hover:text-red-500 transition-opacity p-0.5 mt-0.5 flex-shrink-0"
                        title="删除"
                      >
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
            {savedHistory.length > 0 && (
              <div className="px-4 py-3 border-t border-[var(--border-color)]">
                <button
                  onClick={() => {
                    localStorage.removeItem(HISTORY_KEY);
                    setSavedHistory([]);
                  }}
                  className="text-xs text-red-500 hover:text-red-700 transition-colors"
                >
                  清空全部历史
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Selection Modal */}
      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProvider}
        setProvider={setSelectedProvider}
        model={selectedModel}
        setModel={setSelectedModel}
        isCustomModel={isCustomSelectedModel}
        setIsCustomModel={setIsCustomSelectedModel}
        customModel={customSelectedModel}
        setCustomModel={setCustomSelectedModel}
        isComprehensiveView={isComprehensiveView}
        setIsComprehensiveView={setIsComprehensiveView}
        showFileFilters={false}
        onApply={() => {
          console.log('Model selection applied:', selectedProvider, selectedModel);
        }}
        showWikiType={false}
        authRequired={false}
        isAuthLoading={false}
      />
    </div>
  );
};

export default Ask;
