import { NextRequest, NextResponse } from 'next/server';

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = req.nextUrl;
    const targetUrl = `${TARGET_SERVER_BASE_URL}/bitbucket/repo-structure?${searchParams.toString()}`;

    const res = await fetch(targetUrl);
    const text = await res.text();
    if (!res.ok) {
      return NextResponse.json({ error: text }, { status: res.status });
    }
    return NextResponse.json(JSON.parse(text));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal error' },
      { status: 500 }
    );
  }
}
