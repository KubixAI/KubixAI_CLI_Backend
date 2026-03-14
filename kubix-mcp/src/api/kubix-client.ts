import { KUBIX_API_URL, DASHBOARD_URL, FETCH_TIMEOUT_MS } from '../constants.js';
import { getAccessToken } from '../auth/oauth-client.js';
import { consumeSSEStream } from './sse-consumer.js';
import type {
  OptimizeRequest,
  GenerateRequest,
  StreamingResponse,
  JobStatus,
  CreditsResponse,
  KubixSession,
  SSEEvent,
} from '../types.js';

function fetchWithTimeout(
  url: string,
  init?: RequestInit,
  timeoutMs = FETCH_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  const mergedSignal = init?.signal
    ? AbortSignal.any([controller.signal, init.signal])
    : controller.signal;

  return fetch(url, { ...init, signal: mergedSignal }).finally(() =>
    clearTimeout(timeout)
  );
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await getAccessToken();
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`,
  };
}

// ============================================
// Optimization & Generation
// ============================================

export async function startOptimization(
  request: OptimizeRequest
): Promise<StreamingResponse> {
  const headers = await authHeaders();
  const response = await fetchWithTimeout(
    `${KUBIX_API_URL}/api/v1/kubix/optimize/streaming`,
    {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to start optimization' }));
    throw new Error(
      (error as Record<string, string>).detail ||
        `HTTP ${response.status}: Failed to start optimization`
    );
  }

  return (await response.json()) as StreamingResponse;
}

export async function startGeneration(
  request: GenerateRequest
): Promise<StreamingResponse> {
  const headers = await authHeaders();
  const response = await fetchWithTimeout(
    `${KUBIX_API_URL}/api/v1/kubix/generate/streaming`,
    {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to start generation' }));
    throw new Error(
      (error as Record<string, string>).detail ||
        `HTTP ${response.status}: Failed to start generation`
    );
  }

  return (await response.json()) as StreamingResponse;
}

// ============================================
// SSE Streaming
// ============================================

export async function streamOptimization(
  sessionId: string,
  onEvent?: (event: SSEEvent) => void
): Promise<SSEEvent[]> {
  const token = await getAccessToken();
  const url = `${KUBIX_API_URL}/api/v1/kubix/events/${sessionId}`;
  return consumeSSEStream(url, token, onEvent);
}

// ============================================
// Job Management
// ============================================

export async function getJobStatus(sessionId: string): Promise<JobStatus> {
  const headers = await authHeaders();
  const response = await fetchWithTimeout(
    `${KUBIX_API_URL}/api/v1/kubix/jobs/${sessionId}`,
    { headers }
  );

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: Failed to get job status`);
  }

  return (await response.json()) as JobStatus;
}

export async function cancelJob(sessionId: string): Promise<void> {
  const headers = await authHeaders();
  const response = await fetchWithTimeout(
    `${KUBIX_API_URL}/api/v1/kubix/jobs/${sessionId}`,
    { method: 'DELETE', headers }
  );

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: Failed to cancel job`);
  }
}

// ============================================
// Credits
// ============================================

export async function getCredits(): Promise<CreditsResponse> {
  const token = await getAccessToken();
  const response = await fetchWithTimeout(`${DASHBOARD_URL}/api/kubix/credits`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    if (response.status === 401) {
      throw new Error('Not authenticated. Use kubix_auth to sign in first.');
    }
    throw new Error(`HTTP ${response.status}: Failed to fetch credits`);
  }

  return (await response.json()) as CreditsResponse;
}

// ============================================
// Sessions
// ============================================

export async function getSessions(
  limit = 10,
  status = 'all'
): Promise<{ sessions: KubixSession[]; total: number; has_more: boolean }> {
  const token = await getAccessToken();
  const url = new URL('/api/kubix/sessions', DASHBOARD_URL);
  url.searchParams.set('limit', String(limit));
  if (status !== 'all') url.searchParams.set('status', status);

  const response = await fetchWithTimeout(url.toString(), {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    if (response.status === 401) {
      throw new Error('Not authenticated. Use kubix_auth to sign in first.');
    }
    throw new Error(`HTTP ${response.status}: Failed to fetch sessions`);
  }

  return (await response.json()) as {
    sessions: KubixSession[];
    total: number;
    has_more: boolean;
  };
}

// ============================================
// Health
// ============================================

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetchWithTimeout(`${KUBIX_API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
