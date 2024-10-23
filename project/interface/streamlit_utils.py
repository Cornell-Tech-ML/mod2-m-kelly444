import inspect
import streamlit as st
from typing import Callable, Optional, Any

img_id_counter: int = 0


def get_image_id() -> int:
    global img_id_counter
    img_id_counter += 1
    return img_id_counter


def get_img_tag(src: str, width: Optional[int] = None) -> str:
    img_id = get_image_id()
    if width is not None:
        style = f"""
<style>.img-{img_id} {{
    float: left;
    width: {width}px;
}}
</style>
        """
    else:
        style = ""
    return f"""
        <img src="{src}" class="img-{img_id}" alt="img-{img_id}" />
        {style}
    """


def render_function(fn: Callable[..., Any]) -> None:
    st.markdown(
        f"""
```python
{inspect.getsource(fn)}

```"""
    )
