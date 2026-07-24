# Eyeball what the model generates on arithmetic vs how we grade it.
from chat_cli import device, generate_reply, load_model
from tasks import make_task
from tokenizer import ChatTokenizer

model = load_model("log_sft_d26/model_step_1999.pt")
tok = ChatTokenizer()
dt = "cuda" if device.startswith("cuda") else device
task = make_task("arithmetic", "test")
for i in range(8):
    ex = task[i]
    q = ex["messages"][0]["content"]
    out = generate_reply(model, tok, [{"role": "user", "content": q}],
                         device=device, device_type=dt, max_new_tokens=64, temperature=1.0)
    print(f"\nQ: {q}\nGOLD: {ex['answer']}\nGEN:  {out!r}\n"
          f"SCORE(lenient): {task.evaluate(ex, out, lenient=True)}  "
          f"SCORE(strict): {task.evaluate(ex, out, lenient=False)}")
