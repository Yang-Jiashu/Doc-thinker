export type AutoRotationPauseReason =
  | "disabled"
  | "document-hidden"
  | "layout-running"
  | "selection"
  | "pointer-outside"
  | "pointer-down"
  | "node-drag"
  | "orbit-interaction"
  | "wheel"
  | "resume-delay";

export interface AutoRotationState {
  enabled: boolean;
  pointerInside: boolean;
  pointerDown: boolean;
  orbitInteracting: boolean;
  nodeDragging: boolean;
  wheelActive: boolean;
  hasSelection: boolean;
  layoutStable: boolean;
  documentVisible: boolean;
  lastInteractionAt: number;
}

export interface AutoRotationConfig {
  enabledByDefault: boolean;
  angularSpeedRadPerSecond: number;
  resumeDelayMs: number;
  wheelIdleMs: number;
  pauseWhenPointerOutside: boolean;
  pauseWhenDocumentHidden: boolean;
  pauseWhileLayoutRunning: boolean;
}

export interface AutoRotationDecision {
  shouldRotate: boolean;
  pausedReason: AutoRotationPauseReason | null;
}

export const AUTO_ROTATION_CONFIG: AutoRotationConfig = {
  enabledByDefault: true,
  angularSpeedRadPerSecond: Math.PI * 2 / 45,
  resumeDelayMs: 1_200,
  wheelIdleMs: 180,
  pauseWhenPointerOutside: false,
  pauseWhenDocumentHidden: true,
  pauseWhileLayoutRunning: true,
};

export class AutoRotationController {
  readonly config: AutoRotationConfig;
  private state: AutoRotationState;
  private wheelActiveUntil = 0;

  constructor(
    config: Partial<AutoRotationConfig> = {},
    enabled?: boolean,
    now = performance.now(),
  ) {
    this.config = { ...AUTO_ROTATION_CONFIG, ...config };
    this.state = {
      enabled: enabled ?? this.config.enabledByDefault,
      pointerInside: false,
      pointerDown: false,
      orbitInteracting: false,
      nodeDragging: false,
      wheelActive: false,
      hasSelection: false,
      layoutStable: false,
      documentVisible: true,
      lastInteractionAt: now,
    };
  }

  get snapshot(): Readonly<AutoRotationState> {
    return { ...this.state };
  }

  setEnabled(enabled: boolean, now = performance.now()): void {
    if (this.state.enabled === enabled) return;
    this.state.enabled = enabled;
    this.markInteraction(now);
  }

  setPointerInside(pointerInside: boolean, now = performance.now()): void {
    if (this.state.pointerInside === pointerInside) return;
    this.state.pointerInside = pointerInside;
    this.markInteraction(now);
  }

  setPointerDown(pointerDown: boolean, now = performance.now()): void {
    if (this.state.pointerDown === pointerDown) return;
    this.state.pointerDown = pointerDown;
    this.markInteraction(now);
  }

  setOrbitInteracting(orbitInteracting: boolean, now = performance.now()): void {
    if (this.state.orbitInteracting === orbitInteracting) return;
    this.state.orbitInteracting = orbitInteracting;
    this.markInteraction(now);
  }

  setNodeDragging(nodeDragging: boolean, now = performance.now()): void {
    if (this.state.nodeDragging === nodeDragging) return;
    this.state.nodeDragging = nodeDragging;
    this.markInteraction(now);
  }

  noteWheel(now = performance.now()): void {
    this.state.wheelActive = true;
    this.wheelActiveUntil = now + this.config.wheelIdleMs;
    this.markInteraction(now);
  }

  setHasSelection(hasSelection: boolean, now = performance.now()): void {
    if (this.state.hasSelection === hasSelection) return;
    this.state.hasSelection = hasSelection;
    this.markInteraction(now);
  }

  setLayoutStable(layoutStable: boolean, now = performance.now()): void {
    if (this.state.layoutStable === layoutStable) return;
    this.state.layoutStable = layoutStable;
    this.markInteraction(now);
  }

  setDocumentVisible(documentVisible: boolean, now = performance.now()): void {
    if (this.state.documentVisible === documentVisible) return;
    this.state.documentVisible = documentVisible;
    this.markInteraction(now);
  }

  markInteraction(now = performance.now()): void {
    this.state.lastInteractionAt = now;
  }

  evaluate(now = performance.now()): AutoRotationDecision {
    if (this.state.wheelActive && now >= this.wheelActiveUntil) this.state.wheelActive = false;
    const pausedReason = this.pauseReason(now);
    return { shouldRotate: pausedReason === null, pausedReason };
  }

  private pauseReason(now: number): AutoRotationPauseReason | null {
    if (!this.state.enabled) return "disabled";
    if (this.config.pauseWhenDocumentHidden && !this.state.documentVisible) return "document-hidden";
    if (this.config.pauseWhileLayoutRunning && !this.state.layoutStable) return "layout-running";
    if (this.state.hasSelection) return "selection";
    if (this.config.pauseWhenPointerOutside && !this.state.pointerInside) return "pointer-outside";
    if (this.state.pointerDown) return "pointer-down";
    if (this.state.nodeDragging) return "node-drag";
    if (this.state.orbitInteracting) return "orbit-interaction";
    if (this.state.wheelActive) return "wheel";
    if (now - this.state.lastInteractionAt < this.config.resumeDelayMs) return "resume-delay";
    return null;
  }
}
