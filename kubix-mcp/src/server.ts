import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

// Tools
import { registerKubixAuth } from './tools/forge-auth.js';
import { registerKubixCredits } from './tools/forge-credits.js';
import { registerKubixOptimize } from './tools/forge-optimize.js';
import { registerKubixGenerate } from './tools/forge-generate.js';
import { registerKubixStatus } from './tools/forge-status.js';
import { registerKubixCancel } from './tools/forge-cancel.js';
import { registerKubixSessions } from './tools/forge-sessions.js';

// Resources
import { registerAuthStatusResource } from './resources/auth-status.js';
import { registerCreditsResource } from './resources/credits.js';

// Prompts
import { registerKubixOptimizePrompt } from './prompts/forge-optimize.js';
import { registerKubixAnalyzePrompt } from './prompts/forge-analyze.js';

export function createServer(): McpServer {
  const server = new McpServer({
    name: 'kubix',
    version: '1.0.0',
  });

  // Register tools
  registerKubixAuth(server);
  registerKubixCredits(server);
  registerKubixOptimize(server);
  registerKubixGenerate(server);
  registerKubixStatus(server);
  registerKubixCancel(server);
  registerKubixSessions(server);

  // Register resources
  registerAuthStatusResource(server);
  registerCreditsResource(server);

  // Register prompts
  registerKubixOptimizePrompt(server);
  registerKubixAnalyzePrompt(server);

  return server;
}
