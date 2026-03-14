import * as crypto from 'node:crypto';
import {
  DASHBOARD_URL,
  CLIENT_ID,
  REDIRECT_URI,
  SCOPES,
  OAUTH_POLL_INTERVAL_MS,
  OAUTH_POLL_TIMEOUT_MS,
  TOKEN_REFRESH_THRESHOLD_MS,
  FETCH_TIMEOUT_MS,
} from '../constants.js';
import { loadTokens, saveTokens, clearTokens } from './token-store.js';
import type { OAuthTokens, CreditsResponse } from '../types.js';

// ============================================
// PKCE Helpers
// ============================================

function generateCodeVerifier(): string {
  return crypto.randomBytes(32).toString('base64url');
}

function generateCodeChallenge(verifier: string): string {
  return crypto.createHash('sha256').update(verifier).digest('base64url');
}

function generateState(): string {
  return crypto.randomBytes(16).toString('hex');
}

// ============================================
// Fetch with timeout (for non-SSE calls)
// ============================================

function fetchWithTimeout(
  url: string,
  init?: RequestInit,
  timeoutMs = FETCH_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...init, signal: controller.signal }).finally(() =>
    clearTimeout(timeout)
  );
}

// ============================================
// OAuth Flow
// ============================================

/**
 * Start the full PKCE OAuth flow:
 * 1. Generate PKCE params
 * 2. Open browser to authorize page
 * 3. Poll /api/oauth/check for auth code
 * 4. Exchange code for tokens
 */
export async function startOAuthFlow(): Promise<OAuthTokens> {
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = generateCodeChallenge(codeVerifier);
  const state = generateState();

  // Build authorization URL
  const authUrl = new URL('/oauth/authorize', DASHBOARD_URL);
  authUrl.searchParams.set('response_type', 'code');
  authUrl.searchParams.set('client_id', CLIENT_ID);
  authUrl.searchParams.set('redirect_uri', REDIRECT_URI);
  authUrl.searchParams.set('scope', SCOPES);
  authUrl.searchParams.set('state', state);
  authUrl.searchParams.set('code_challenge', codeChallenge);
  authUrl.searchParams.set('code_challenge_method', 'S256');

  // Open browser (dynamic import for ESM-only package)
  const open = (await import('open')).default;
  await open(authUrl.toString());

  // Poll for authorization code
  const code = await pollForAuthCode(state);

  // Exchange code for tokens
  const tokens = await exchangeCodeForTokens(code, codeVerifier);
  saveTokens(tokens);
  return tokens;
}

async function pollForAuthCode(state: string): Promise<string> {
  const checkUrl = new URL('/api/oauth/check', DASHBOARD_URL);
  checkUrl.searchParams.set('state', state);
  checkUrl.searchParams.set('client_id', CLIENT_ID);

  const deadline = Date.now() + OAUTH_POLL_TIMEOUT_MS;

  while (Date.now() < deadline) {
    try {
      const response = await fetchWithTimeout(checkUrl.toString());
      if (response.ok) {
        const data = (await response.json()) as {
          completed: boolean;
          code?: string;
          error?: string;
        };
        if (data.completed && data.code) {
          return data.code;
        }
        if (data.error) {
          throw new Error(`Authorization failed: ${data.error}`);
        }
      }
    } catch (err) {
      if (err instanceof Error && err.message.startsWith('Authorization failed')) {
        throw err;
      }
      // Network error, keep polling
    }

    await new Promise((r) => setTimeout(r, OAUTH_POLL_INTERVAL_MS));
  }

  throw new Error(
    'Authorization timed out. Please try again. If the browser did not open, visit: ' +
      DASHBOARD_URL +
      '/oauth/authorize'
  );
}

async function exchangeCodeForTokens(
  code: string,
  codeVerifier: string
): Promise<OAuthTokens> {
  const tokenUrl = new URL('/api/oauth/token', DASHBOARD_URL);

  const body = new URLSearchParams();
  body.set('grant_type', 'authorization_code');
  body.set('client_id', CLIENT_ID);
  body.set('code', code);
  body.set('code_verifier', codeVerifier);
  body.set('redirect_uri', REDIRECT_URI);

  const response = await fetchWithTimeout(tokenUrl.toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });

  if (!response.ok) {
    const error = (await response.json().catch(() => ({}))) as Record<string, string>;
    throw new Error(
      `Token exchange failed: ${error.error_description || error.error || response.statusText}`
    );
  }

  const tokens = (await response.json()) as OAuthTokens;
  tokens.expires_at = Date.now() + tokens.expires_in * 1000;
  return tokens;
}

// ============================================
// Token Refresh
// ============================================

export async function refreshTokens(tokens: OAuthTokens): Promise<OAuthTokens> {
  const tokenUrl = new URL('/api/oauth/token', DASHBOARD_URL);

  const body = new URLSearchParams();
  body.set('grant_type', 'refresh_token');
  body.set('client_id', CLIENT_ID);
  body.set('refresh_token', tokens.refresh_token);

  const response = await fetchWithTimeout(tokenUrl.toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });

  if (!response.ok) {
    clearTokens();
    throw new Error('Token refresh failed. Please re-authenticate with kubix_auth.');
  }

  const newTokens = (await response.json()) as OAuthTokens;
  newTokens.expires_at = Date.now() + newTokens.expires_in * 1000;
  saveTokens(newTokens);
  return newTokens;
}

// ============================================
// Token Access (auto-refresh with mutex)
// ============================================

// Singleton promise prevents concurrent refresh races
let refreshInFlight: Promise<OAuthTokens> | null = null;

/**
 * Get a valid access token, refreshing if necessary.
 * Uses a mutex so concurrent callers don't trigger parallel refreshes.
 * Throws if no tokens are stored (user needs to authenticate).
 */
export async function getAccessToken(): Promise<string> {
  let tokens = loadTokens();
  if (!tokens) {
    throw new Error('Not authenticated. Use the kubix_auth tool to sign in first.');
  }

  // Refresh if within threshold of expiry
  if (Date.now() + TOKEN_REFRESH_THRESHOLD_MS >= tokens.expires_at) {
    if (!refreshInFlight) {
      refreshInFlight = refreshTokens(tokens).finally(() => {
        refreshInFlight = null;
      });
    }
    tokens = await refreshInFlight;
  }

  return tokens.access_token;
}

/**
 * Validate stored tokens by making a test API call.
 * Returns credits response on success, null on failure.
 */
export async function validateTokens(): Promise<CreditsResponse | null> {
  try {
    const token = await getAccessToken();
    const creditsUrl = new URL('/api/kubix/credits', DASHBOARD_URL);
    const response = await fetchWithTimeout(creditsUrl.toString(), {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) return null;
    return (await response.json()) as CreditsResponse;
  } catch {
    return null;
  }
}
