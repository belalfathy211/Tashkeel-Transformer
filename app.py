import gradio as gr
import torch
from transformer.tashkeel_model import TashkeelModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('tashkeel_transformer.pth', map_location=device)

model = TashkeelModel(
    n_embd=checkpoint.get('n_embd', 384),
    n_head=checkpoint.get('n_head', 6),
    block_size=checkpoint.get('block_size', 256),
    n_layer=checkpoint.get('n_layer', 6),
    tashkeel_vocab_size=checkpoint['tashkeel_vocab_size'],
    chars_vocab_size=checkpoint['chars_vocab_size'],
    dropout=0.0
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def diacritize_text(input_text):
    if not input_text.strip():
        return ""

    tashkeelat = {'َ', 'ُ', 'ِ', 'ً', 'ٌ', 'ٍ', 'ْ', 'ّ', 'َّ', 'ُّ', 'ِّ', 'ًّ', 'ٌّ', 'ٍّ'}
    cleaned_text = "".join([ch for ch in input_text if ch not in tashkeelat])[:256]

    unk_id = checkpoint['stoi_chars'].get('<UNK>', 0)
    input_ids = [checkpoint['stoi_chars'].get(ch, unk_id) for ch in cleaned_text]
    start_seq = [checkpoint['stoi_tashkeel']["<START>"]]

    out_ids = model.predict(input_ids, start_seq)
    tashkeel_list = [checkpoint['itos_tashkeel'].get(t_id, "") for t_id in out_ids]

    out = []
    for i, ch in enumerate(cleaned_text):
        out.append(ch)
        if i < len(tashkeel_list) and tashkeel_list[i] not in ["", "<UNK>", "<END>", "<START>"]:
            out.append(tashkeel_list[i])
    return "".join(out)


demo = gr.Interface(
    fn=diacritize_text,
    inputs=gr.Textbox(lines=3, placeholder="أدخل النص العربي بدون تشكيل هنا...", label="Input Text (Without Tashkeel)"),
    outputs=gr.Textbox(lines=3, label="Diacritized Output (النتيجة بالتشكيل)"),
    title="📖 Sibawayh Al-Ghalaba (سيبويه الغلابة)",
    description="An Encoder-Decoder Transformer built from scratch in PyTorch for Arabic Text Diacritization.",
    examples=[
        [" تعلموا العربية؛ فإنها من دينكم - الفاروق رضي الله عنه"],
        ["من أحب أن يفتح الله قلبه أو يرزقه علماً فعليه بالخلوة، وقلة الأكل، وترك مخالطة السفهاء -الإمام الشافعي"],
        ["ليس العلم بكثرة الرواية، إنما العلم نور يجعله الله في القلب -الإمام مالك"]
    ]
)

if __name__ == "__main__":
    demo.launch()