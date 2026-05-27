import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "local_storage",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
)

def local_storage(key: str, value: str = "") -> str:
    """Read/write a value from browser localStorage.
    
    On first render, returns the stored value (or empty string).
    When `value` changes from Python, writes it to localStorage and returns it.
    """
    return _component_func(key=key, value=value, default="")
