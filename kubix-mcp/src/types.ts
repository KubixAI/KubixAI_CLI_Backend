// ============================================
// Error Sanitization
// ============================================

const TOKEN_PATTERNS = [
  /Bearer\s+[A-Za-z0-9\-._~+/]+=*/gi,
  /eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_.+/=]*/g, // JWTs
  /[a-f0-9]{40,}/gi, // hex tokens
  /refresh_token=[^\s&]+/gi,
  /access_token=[^\s&]+/gi,
];

export function sanitizeError(err: unknown): string {
  let msg = err instanceof Error ? err.message : String(err);
  for (const pattern of TOKEN_PATTERNS) {
    msg = msg.replace(pattern, '[REDACTED]');
  }
  return msg;
}

// ============================================
// Type Guards for SSE Event Data
// ============================================

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function isIterationCompleteData(d: unknown): d is IterationCompleteData {
  return isRecord(d) && typeof d.iteration === 'number' && typeof d.speedup === 'number';
}

export function isSessionCompleteData(d: unknown): d is SessionCompleteData {
  return isRecord(d) && typeof d.best_speedup === 'number';
}

export function isErrorData(d: unknown): d is ErrorData {
  return isRecord(d) && typeof d.message === 'string';
}

// ============================================
// OAuth Types
// ============================================

export interface OAuthTokens {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token: string;
  scope: string;
  expires_at: number; // Absolute timestamp (Date.now() based)
}

export interface CreditsResponse {
  email: string;
  user_id: string;
  balance: number;
  total_purchased: number;
  total_used: number;
  plan_type: string;
  monthly_bonus: number;
  monthly_bonus_claimed: boolean;
}

// ============================================
// Kubix API Request Types
// ============================================

export type HardwareTarget = 'B200' | 'H200' | 'H100' | 'L40S' | 'L4' | 'A100' | 'A10' | 'T4';
export type OutputFormat = 'triton' | 'native_cuda';

export interface OptimizeRequest {
  pytorch_code: string;
  kernel_name: string;
  output_format: OutputFormat;
  target_speedup: number;
  max_iterations: number;
  gpu: HardwareTarget;
  user_prompt?: string;
}

export interface GenerateRequest {
  operation: string;
  description: string;
  input_shapes: number[][];
  output_shape?: number[];
  dtype?: string;
  output_format: OutputFormat;
  target_speedup: number;
  max_iterations: number;
  gpu: HardwareTarget;
  user_prompt?: string;
}

// ============================================
// Kubix API Response Types
// ============================================

export interface StreamingResponse {
  session_id: string;
  stream_url: string;
  status: 'started';
}

export interface JobStatus {
  session_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_iteration: number;
  max_iterations: number;
  best_speedup: number;
  best_iteration: number;
  error?: string;
}

// ============================================
// SSE Event Types
// ============================================

export type SSEEventType =
  | 'session_start'
  | 'thinking'
  | 'token'
  | 'compile_start'
  | 'compile_result'
  | 'profile_start'
  | 'profile_result'
  | 'test_start'
  | 'test_result'
  | 'iteration_start'
  | 'iteration_complete'
  | 'session_complete'
  | 'error'
  | 'heartbeat'
  | 'agent_spawn'
  | 'agent_complete'
  | 'rag_query'
  | 'rag_result'
  | 'root_cause'
  | 'fix_hint'
  | 'adaptive_temp'
  | 'learning'
  | 'kernel_evaluated'
  | 'metric';

export interface SSEEvent {
  id: string;
  type: SSEEventType;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface IterationCompleteData {
  iteration: number;
  speedup: number;
  best_speedup: number;
  correct: boolean;
  code?: string;
  is_best?: boolean;
  latency_ms?: number;
  occupancy?: number;
  bandwidth_util?: number;
  bottleneck?: string;
  memory_throughput_gbps?: number;
  compute_throughput_tflops?: number;
  shared_memory_usage_bytes?: number;
  register_usage_per_thread?: number;
  sm_efficiency?: number;
  warp_execution_efficiency?: number;
  l1_hit_rate?: number;
  l2_hit_rate?: number;
}

export interface SessionCompleteData {
  session_id?: string;
  best_speedup: number;
  best_iteration: number;
  latency_before?: number;
  latency_after?: number;
  optimized_code: string;
  credits_used?: number;
  termination_reason?: string;
  total_iterations?: number;
  success?: boolean;
  gpu?: string;
  output_format?: string;
  all_kernels?: Record<string, unknown>[];
}

export interface ErrorData {
  message: string;
  error_type?: string;
  details?: string;
}

// ============================================
// Sessions Types
// ============================================

export interface KubixSession {
  id: string;
  session_type: string;
  task_name: string;
  gpu_type: string | null;
  backend: string;
  output_format: string;
  target_speedup: number;
  best_speedup: number;
  generations_run: number;
  credits_used: number;
  status: string;
  error_message: string | null;
  multi_gpu: boolean;
  target_gpus: string[] | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  created_at: string;
}
