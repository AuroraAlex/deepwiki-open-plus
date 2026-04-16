import { NextRequest, NextResponse } from 'next/server';

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = req.nextUrl;
    const targetUrl = `${TARGET_SERVER_BASE_URL}/file/content?${searchParams.toString()}`;

    const res = await fetch(targetUrl);
    if (!res.ok) {
      const err = await res.text();
      return NextResponse.json({ error: err }, { status: res.status });
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal error' },
      { status: 500 }
    );
  }
}
