export const DASHBOARD_URL = 'https://dashboard.kubixai.co';
export const KUBIX_API_URL = 'https://api.kubix.ai';
export const CLIENT_ID = 'kubix-kubix-cli';
export const REDIRECT_URI = `${DASHBOARD_URL}/oauth/success`;
export const SCOPES = 'read write llm:agent llm:completion';

export const TOKEN_FILE_DIR = '.kubix';
export const TOKEN_FILE_NAME = 'tokens.json';

export const OAUTH_POLL_INTERVAL_MS = 2000;
export const OAUTH_POLL_TIMEOUT_MS = 300_000; // 5 minutes
export const TOKEN_REFRESH_THRESHOLD_MS = 5 * 60 * 1000; // 5 minutes before expiry

export const SSE_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes max for optimization
export const FETCH_TIMEOUT_MS = 30_000; // 30 seconds for non-SSE requests

export const GPU_TYPES = ['B200', 'H200', 'H100', 'L40S', 'L4', 'A100', 'A10', 'T4'] as const;
export const OUTPUT_FORMATS = ['triton', 'native_cuda'] as const;

export const MAX_PYTORCH_CODE_BYTES = 512_000; // 500 KB
export const MAX_USER_PROMPT_BYTES = 10_240; // 10 KB
