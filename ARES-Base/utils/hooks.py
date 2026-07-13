#this is a conceptual desgin for ARES Hook system 
from typing import Dict, Any, Callable, List, Optional
import torch

class HookRegistry:
    """
    Centralized event dispatcher for ARES research modules.
    Allows external frameworks (RSVLM, Drift Detectors, Explainability probes) 
    to inspect internal activations and attention distributions without modifying model code.
    """
    def __init__(self):
        # Pre-defined research lifecycle event hooks
        self._listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {
            "before_attn": [],
            "after_attn": [],
            "after_ffn": [],
            "after_block": [],
            "after_logits": []
        }
        self.enabled: bool = True

    def register(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribes an external callback function to a specific model event."""
        if event_name not in self._listeners:
            raise ValueError(f"Unknown hook event: '{event_name}'. Available: {list(self._listeners.keys())}")
        self._listeners[event_name].append(callback)
        print(f"[HookRegistry] Registered probe to event: '{event_name}'")

    def dispatch(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Dispatches internal tensor payloads to all subscribed research probes."""
        if not self.enabled or event_name not in self._listeners:
            return
            
        for callback in self._listeners[event_name]:
            # Pass data to external probes (e.g., RSVLM reliability calculators)
            callback(payload)

    def clear_hooks(self, event_name: Optional[str] = None) -> None:
        """Removes registered hooks to prevent memory leaks between experiments."""
        if event_name:
            if event_name in self._listeners:
                self._listeners[event_name].clear()
        else:
            for key in self._listeners:
                self._listeners[key].clear()
        print("[HookRegistry] Cleared active hooks.")