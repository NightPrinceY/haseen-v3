"""V3 agent set: replies of open-weight LLMs asked to answer a customer "in dialect T" (the failure mode HaSeen
monitors). Runs on a GPU sandbox with transformers; output: agent_raw.jsonl {model, target, scenario, customer, reply}.
These replies are TEST data only: they are labelled word by word afterwards (Claude with evidence; Egyptian reviewed by
a native speaker) and never enter training."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = sys.argv[1].split(",")
OUT = sys.argv[2]
DIALECT = {"EGY": "باللهجة المصرية العامية فقط", "SAU": "باللهجة السعودية (النجدية) العامية فقط",
           "GLF": "باللهجة الخليجية العامية فقط (كويتية أو إماراتية)", "LEV": "باللهجة الشامية العامية فقط",
           "IRQ": "باللهجة العراقية العامية فقط", "MSA": "بالعربية الفصحى فقط"}
SCEN = [  # (company, customer message): neutral, simple wording; only the agent reply is evaluated
    ("بنك", "بطاقتي اتسحبت مني في ماكينة الصراف، أعمل إيه؟"),
    ("بنك", "عايز أعرف ليه في خصم على حسابي الشهر ده."),
    ("شركة اتصالات", "النت عندي بطيء جدا من يومين."),
    ("شركة اتصالات", "أريد إلغاء الباقة الحالية وتغييرها لباقة أرخص."),
    ("شركة اتصالات", "الفاتورة جاية أعلى من المعتاد."),
    ("تطبيق توصيل طعام", "الطلب اتأخر ساعة ولسه موصلش."),
    ("تطبيق توصيل طعام", "الأكل وصل ناقص صنف."),
    ("متجر إلكتروني", "أبغى أرجع منتج اشتريته الأسبوع الماضي."),
    ("متجر إلكتروني", "متى يوصل طلبي؟ رقم الطلب عندي."),
    ("شركة طيران", "رحلتي اتلغت، هل أقدر أحجز رحلة ثانية؟"),
    ("شركة طيران", "أريد أضيف شنطة زيادة على حجزي."),
    ("عيادة", "أبي أحجز موعد مع دكتور أسنان بكرة."),
    ("عيادة", "هل في موعد متاح بعد العصر؟"),
    ("شركة تأمين", "عملت حادث بسيط بالسيارة، أعمل إيه عشان التأمين؟"),
    ("شركة كهرباء", "الكهرباء مقطوعة عندي من الصبح."),
    ("شركة شحن", "الشحنة واقفة في الجمارك من أسبوع."),
    ("فندق", "أريد أغير تاريخ الحجز."),
    ("بنك", "نسيت الرقم السري للتطبيق."),
    ("متجر إلكتروني", "دفعت مرتين على نفس الطلب."),
    ("شركة اتصالات", "أبي أعرف عروض الإنترنت المتوفرة."),
]
rows = [(T, i, comp, msg) for T in DIALECT for i, (comp, msg) in enumerate(SCEN)]
with open(OUT, "a") as fo:
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        for k in range(0, len(rows), 24):
            batch = rows[k:k + 24]
            prompts = []
            for T, i, comp, msg in batch:
                sys_msg = (f"أنت موظف خدمة عملاء في {comp}. رد على العميل {DIALECT[T]}، في جملتين أو ثلاث جمل، "
                           f"بأسلوب مهذب وعملي. لا تكتب أي شيء غير الرد.")
                conv = [{"role": "user", "content": sys_msg + "\n\nرسالة العميل: " + msg}]
                kw = {"enable_thinking": False} if "Qwen3" in name else {}
                prompts.append(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **kw))
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9,
                                     pad_token_id=tok.pad_token_id)
            for (T, i, comp, msg), o in zip(batch, out):
                reply = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                fo.write(json.dumps({"model": name, "target": T, "scenario": i, "company": comp, "customer": msg,
                                     "reply": reply}, ensure_ascii=False) + "\n")
            fo.flush()
        print(name, "done", flush=True)
        del model
        torch.cuda.empty_cache()
