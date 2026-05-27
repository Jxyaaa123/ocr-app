import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "local_storage",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
)

def local_storage(storage_key: str, value: str = "") -> str:
    return _component_func(storage_key=storage_key, value=value, default="")
