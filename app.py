import gradio as gr


def greet(name, max_tokens=60):
    return f'Hello {name}!!! You will generate {max_tokens} tokens.'


demo = gr.Interface(
    fn=greet,
    inputs=['text', gr.Slider(50, 500)],
    outputs='text'
)

demo.launch()
