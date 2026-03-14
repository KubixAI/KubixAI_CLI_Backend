import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { loadTokens } from '../auth/token-store.js';
import { TOKEN_REFRESH_THRESHOLD_MS } from '../constants.js';

export function registerAuthStatusResource(server: McpServer): void {
  server.registerResource(
    'auth-status',
    'kubix://auth/status',
    {
      title: 'Kubix Authentication Status',
      description: 'Current authentication status with the Kubix service',
      mimeType: 'application/json',
    },
    async (uri) => {
      const tokens = loadTokens();
      const now = Date.now();
      const authenticated =
        tokens !== null && now < tokens.expires_at;
      const needsRefresh =
        tokens !== null &&
        now + TOKEN_REFRESH_THRESHOLD_MS >= tokens.expires_at;

      return {
        contents: [
          {
            uri: uri.href,
            mimeType: 'application/json',
            text: JSON.stringify({
              authenticated,
              needs_refresh: needsRefresh,
              expires_at: tokens?.expires_at
                ? new Date(tokens.expires_at).toISOString()
                : null,
              has_refresh_token: !!tokens?.refresh_token,
              scope: tokens?.scope || null,
            }),
          },
        ],
      };
    }
  );
}
