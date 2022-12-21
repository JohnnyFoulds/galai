import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/galactica-1.3b")
text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, num_workers=2)

def predict(text, max_length=64, temperature=0.7, do_sample=True):
    text = text.strip()
    out_text = text2text_generator(text, max_length=max_length, 
                              temperature=temperature, 
                              do_sample=do_sample,
                              eos_token_id = tokenizer.eos_token_id,
                              bos_token_id = tokenizer.bos_token_id,
                              pad_token_id = tokenizer.pad_token_id,
                         )[0]['generated_text']
    out_text = "<p>" + out_text + "</p>"
    out_text = out_text.replace(text, text + "<b><span style='background-color: #ffffcc;'>")
    out_text = out_text +  "</span></b>"
    out_text = out_text.replace("\n", "<br>")
    return out_text

iface = gr.Interface(
    fn=predict, 
    inputs=[
        gr.inputs.Textbox(lines=5, label="Input Text"),
        gr.inputs.Slider(minimum=32, maximum=256, default=64, label="Max Length"),
        gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.7, step=0.1, label="Temperature"),
        gr.inputs.Checkbox(label="Do Sample"),
    ],
    outputs=gr.HTML(),
    description="Galactica Base Model",
    examples=[[
            "The attention mechanism in LLM is",
            128,
            0.7,
            True
        ], 
        [
            "Title: Attention is all you need\n\nAbstract:",
            128,
            0.7,
            True
        ]
    ]
)

iface.launch()