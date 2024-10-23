from argparse import ArgumentParser
import streamlit as st
from project.interface.streamlit_utils import get_img_tag
from project.interface.train import render_train_interface
from project.math_interface import render_math_sandbox
from project.run_torch import TorchTrain
from typing import Callable, Dict, Any


class MiniTorchApp:
    def __init__(self, module_num: int, hide_function_defs: bool) -> None:
        self.module_num: int = module_num
        self.hide_function_defs: bool = hide_function_defs
        self.pages: Dict[str, Callable[[], None]] = {}

        self.setup_sidebar()

    def setup_sidebar(self) -> None:
        st.set_page_config(page_title="Interactive MiniTorch")
        st.sidebar.markdown(
            f"""
<h1 style="font-size:30pt; float: left; margin-right: 20px; margin-top: 1px;">MiniTorch</h1>{get_img_tag("https://minitorch.github.io/logo-sm.png", width=40)}
""",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            """
    [Documentation](https://minitorch.github.io/)
"""
        )
        module_selection: str = st.sidebar.radio(
            "Module",
            [f"Module {i}" for i in range(self.module_num + 1)],
            index=self.module_num,
        )

        self.load_module_pages(module_selection)

    def load_module_pages(self, module_selection: str) -> None:
        if module_selection == "Module 0":
            self.add_module_0_pages()
        elif module_selection == "Module 1":
            self.add_module_1_pages()
        elif module_selection == "Module 2":
            self.add_module_2_pages()
        elif module_selection == "Module 3":
            self.add_module_3_pages()
        elif module_selection == "Module 4":
            self.add_module_4_pages()

        self.render_selected_page()

    def add_module_0_pages(self) -> None:
        from project.module_interface import render_module_sandbox
        from project.run_manual import ManualTrain

        self.pages["Math Sandbox"] = self.render_math_sandbox(False)
        self.pages["Module Sandbox"] = render_module_sandbox
        self.pages["Torch Example"] = self.render_torch_example()
        self.pages["Module 0: Manual"] = self.render_manual_train()

    def add_module_1_pages(self) -> None:
        from project.run_scalar import ScalarTrain
        from project.show_expression_interface import render_show_expression

        self.pages["Scalar Sandbox"] = self.render_math_sandbox(True)
        self.pages["Autodiff Sandbox"] = render_show_expression
        self.pages["Module 1: Scalar"] = self.render_scalar_train(ScalarTrain)

    def add_module_2_pages(self) -> None:
        from project.run_tensor import TensorTrain
        from project.show_expression_interface import render_show_expression
        from project.tensor_interface import render_tensor_sandbox

        self.pages["Tensor Sandbox"] = (
            self.render_tensor_sandbox()
        )  # no parameters needed
        self.pages["Tensor Math Sandbox"] = self.render_math_sandbox(True, True)
        self.pages["Autograd Sandbox"] = render_show_expression
        self.pages["Module 2: Tensor"] = self.render_tensor_train(TensorTrain)

    def add_module_3_pages(self) -> None:
        from project.run_fast_tensor import FastTrain

        self.pages["Module 3: Efficient"] = self.render_fast_train(FastTrain)

    def add_module_4_pages(self) -> None:
        from project.run_mnist_interface import render_run_image_interface
        from project.sentiment_interface import render_run_sentiment_interface

        self.pages["Module 4: Images"] = render_run_image_interface
        self.pages["Module 4: Sentiment"] = render_run_sentiment_interface

    def render_selected_page(self) -> None:
        page_options: list[str] = list(self.pages.keys())
        page_selection: str = st.sidebar.radio("Pages", page_options)

        if page_selection in self.pages:
            self.pages[page_selection]()  # Call the selected page function

    def render_math_sandbox(self, *args: Any) -> Callable[[], None]:
        st.header("Math Sandbox")
        return lambda: render_math_sandbox(*args)

    def render_tensor_sandbox(self) -> Callable[[], None]:
        from project.tensor_interface import render_tensor_sandbox

        return lambda: render_tensor_sandbox()  # no parameters passed

    def render_scalar_train(self, ScalarTrain: Any) -> Callable[[], None]:
        return lambda: render_train_interface(ScalarTrain)

    def render_tensor_train(self, TensorTrain: Any) -> Callable[[], None]:
        return lambda: render_train_interface(TensorTrain)

    def render_fast_train(self, FastTrain: Any) -> Callable[[], None]:
        return lambda: render_train_interface(FastTrain, False)

    def render_torch_example(self) -> Callable[[], None]:
        return lambda: render_train_interface(TorchTrain, False)

    def render_manual_train(self) -> Callable[[], None]:
        from project.run_manual import ManualTrain

        return lambda: render_train_interface(ManualTrain, False, False, True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("module_num", type=int)
    parser.add_argument(
        "--hide_function_defs", action="store_true", dest="hide_function_defs"
    )
    args = parser.parse_args()

    app = MiniTorchApp(args.module_num, args.hide_function_defs)
