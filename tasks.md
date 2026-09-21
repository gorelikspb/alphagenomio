# Alphag — tasks

Чеклист по [README](README.md) и [vision.md](vision.md). Обновляй по мере прогресса.

## Сейчас в коде / репо

- [x] **Single sequence** — запуск DNase, stats, peaks, JSON
- [x] **Compare ref vs mutant** — два поля, таблица Δ (mean, max, peak shift)
- [x] **BYOK + disclaimer** — ключ в форме, текст «non-commercial / not official Google»
- [x] **README + vision.md** — описан flow и цели проекта
- [x] **Push в GitHub** — compare-режим в `main` (коммит `453cee3`)
- [x] **Live на Render** — Compare работает с реальным ключом

## Дальше (из vision, шаг 2–3)

- [x] **Выбор ткани** — dropdown lung / liver / brain
- [x] **Insert example** — кнопки с GATTACA / GATTACG
- [x] **Download CSV** — экспорт таблицы Δ после Compare
- [ ] **Push + live** — задеплоить CSV (и проверить tissue / examples на Render)

## Следующие шаги (по приоритету)

1. **Push на GitHub / Render** — чтобы на live появилась кнопка Download CSV.
2. **Пауза 1–2 недели** — смотреть, есть ли фидбек на Stack Exchange (вопрос AlphaGenome).
3. **Biology SE** — 1–2 простых ответа без ссылок (карма), не alphag.
4. **Опционально позже:**
   - короткая фраза на сайте: demo / research only;
   - ссылка с seqanalysis.org (related tool), если есть трафик;
   - ввод по координатам (chr + position) — только если кто-то попросит.

**Не сейчас:** AlphaFold, несколько моделей, VCF batch, платный SaaS.

## Критерий «готово для пользователя» (vision)

- [ ] **5–10 человек вне автора** попробовали Compare за 2–3 месяца после live-деплоя

**1. API key** — свой ключ AlphaGenome (не выдумывать):

- Запросить здесь: https://deepmind.google.com/science/alphagenome  
- Вставить в поле **AlphaGenome API key** на сайте. Ключ никуда не сохраняется.

**2. Compare** — или нажми **Insert example**, затем **Compare**.

| Поле | Текст |
|------|--------|
| Reference | `GATTACA` |
| Mutant | `GATTACG` |

**3. Single** — **Insert example** → **Run AlphaGenome**.

**4. Tissue** — выбери Lung / Liver / Brain в dropdown и запусти снова.

## Критерий «готово для пользователя» (vision)

- [ ] **5–10 человек вне автора** попробовали Compare за 2–3 месяца после live-деплоя

---

## Как найти пользователей без коллег (из proteinanalyse)

Тот же канал, что для seqanalysis.org — см. `C:\dev\proteinanalyse\docs\seo\`.

**Сайты:**

- https://bioinformatics.stackexchange.com/ — основной для alphag
- https://biology.stackexchange.com/ — если вопрос про варианты / регуляцию проще

**Правило (не спам):**

1. Сначала **полезный ответ** (2–4 абзаца, как в Colab / docs).
2. Одна ссылка в конце + **Disclosure: I maintain this tool**.
3. Цель: **1–2 ответа в месяц**, не рекламный пост.

**На какие вопросы отвечать:**

- «How to compare reference vs alternate for regulatory effect?»
- «AlphaGenome API — quick way to check variant without Colab?»
- «DNase / chromatin accessibility prediction for a short sequence»

**Профиль** (черновик из proteinanalyse, адаптировать):

> Learning bioinformatics. I build small free research tools, including alphagenomio.onrender.com — a browser UI for AlphaGenome DNase compare (BYOK, non-commercial). Personal capacity.

**Сегодня (15 мин):**

1. Обновить About на Stack Exchange (1–2 предложения).
2. Поиск: `[alphagenome]` или `variant effect prediction online` на bioinformatics.SE.
3. Если есть свежий вопрос — ответить по шаблону ниже. Если нет — **задать свой** (см. proteinanalyse `STACKEXCHANGE_QUESTION_IDEAS.md`).

### Черновик ответа (когда спросят про ref vs mut + AlphaGenome)

> For a quick non-commercial check with your own API key, you can run the same sequence twice (REF and ALT) or use a small compare UI that pads to 1 Mb and reports DNase deltas per track.  
> AlphaGenome docs recommend `predict_variant` for interval-based work; for pasted sequences, compare two `predict_sequence` runs on the same tissue ontology.  
> I maintain a minimal free tool for that: https://alphagenomio.onrender.com/ (Compare ref vs mutant, BYOK).  
> *Disclosure: I maintain this tool.*

---

## Stack Exchange: что может агент, что — только ты

### Могу я (Cursor) сам зайти и написать от твоего имени?

**Нет.** У меня нет твоего логина на Biology SE.

| Могу | Не могу |
|------|---------|
| Читать **публичные** вопросы (API, web) | Логиниться как ты |
| Искать темы (`alphagenome`, variant effect) | Нажимать Post / Answer |
| Написать **черновик** ответа или вопроса | Постить от твоего аккаунта |

**Проверил сейчас:** на Biology и Bioinformatics SE **нет вопросов с «alphagenome»** — отвечать некому. Логичнее **задать свой вопрос** на Bioinformatics SE (черновик ниже).

### Что в PM (`C:\dev\PM`)

| Есть | Нет |
|------|-----|
| Telegram-бот + «команда» (Секретарь, PM, Кодер, Техник) через Gemini | Интеграции с Stack Exchange |
| Сканер проектов, советник, dispatch Cursor | Браузер / Playwright |
| `PROMO.md` — тексты для Reddit/чатов | Автопостинг на форумы |

В `PROMO.md` прямо: **посты в чаты кидаешь только ты** — та же логика для SE.

**Grok Bot в PM** — это чат ролей в Telegram, не бот, который заходит на сайты.

### Как сделать «почти автоматом» (реалистично)

1. **Сейчас (без кода):** я пишу черновик → ты копируешь → Post (2 мин).
2. **Позже в PM:** скрипт `stackexchange-scan.js` — API ищет вопросы → Gemini черновик → тебе в Telegram «вот ответ, вставь».
3. **Полный автопост:** OAuth SE API + риск бана за spam — **не советую**.

### Куда постить что (два аккаунта SE)

| Проект | Сайт | Почему |
|--------|------|--------|
| **seqanalysis.org** (FASTA, перевод) | **Biology SE** — больше кармы (~106) | Уже есть вопрос про FASTQ |
| **alphag** (AlphaGenome compare) | **Bioinformatics SE** — карма ~21 | Тема ближе к API/пайплайнам |

---

## Скопируй и опубликуй (Bioinformatics SE — alphag)

### Шаг 1 — About (профиль, мягко)

**Куда:** https://bioinformatics.stackexchange.com/users/edit/profile → **About me**

**Текст (без ссылок — не выглядит как реклама):**

```
Learning bioinformatics and building small browser tools for routine sequence checks. Not a professional biologist — personal learning projects.
```

**Website** (отдельное поле, если есть): `https://seqanalysis.org` — одна ссылка достаточно; alphag можно не указывать.

---

### Шаг 2 — Задать вопрос (честный, без ссылки)

**Куда:** https://bioinformatics.stackexchange.com/questions/ask

Это **не подстава**, если вопрос реальный — ты правда это делал, когда писал alphag. Так делают на SE.

**Title (широко — не «я уже выбрал DNase»):**

```
AlphaGenome API: practical workflow to compare reference vs alternate sequences?
```

**Tags (до 5, вводи по одному, сайт подскажет):**

| Тег | Зачем |
|-----|--------|
| `api` | про AlphaGenome API |
| `python` | SDK на Python |
| `variants` | ref vs mutated |
| `dna` | вставленные последовательности |

Пятый по желанию: `machine-learning` (AlphaGenome = модель).

**Нет тега `alphagenome`** на сайте, не выдумывай. `variant-calling` можно, но это чаще про GATK/VCF, для твоего вопроса мягче `variants`.

**Body:**

```
I have an AlphaGenome API key for non-commercial use. I want to compare a reference DNA sequence and a mutated version (same length, pasted as text).

The docs show predict_sequence and predict_variant. For two equal-length strings, what do people usually do?

1. Run predict_sequence twice with the same settings and compare the results by hand?
2. Or use predict_variant with genomic coordinates even for small test sequences?

I am still learning which output types are useful for a simple variant check (regulatory tracks, accessibility, etc.). I want a simple workflow I can repeat without opening Colab every time.
```

**Почему так:** в заголовке — **общий** вопрос про AlphaGenome + ref/alt. DNase только в теле («e.g. accessibility») — как у человека, который **ещё разбирается**, а не уже заточен под один трек.

**Не планируй сразу отвечать сам себе со ссылкой** — см. ниже.

---

### «Сам спросил — сам ответил» — тупо?

**Может так выглядеть**, если через день свой ответ со ссылкой на alphag. Модераторы и люди это не любят.

| Нормально | Подозрительно |
|-----------|----------------|
| Спросил → **ждёшь** чужие ответы | Спросил → через час свой ответ + ссылка |
| Через **2+ недели** тишины — коротко «what worked for me» + Disclosure | Вопрос написан только ради ссылки |
| Ссылка только в **профиле** | Каждый пост — твой сайт |

**Реалистичнее для alphag:**

1. **About** — мягкий (выше).  
2. **Вопрос** — опубликовать, **не** отвечать самому с alphag.  
3. **Ждать** — может кто-то ответит (AlphaGenome новый, может тишина).  
4. **Ссылку на alphag** — не гнаться; профиля и демо-сайта достаточно, если кто-то зайдёт в профиль.

**Проще и не «подстава»:** на **Biology SE** второй ответ к **старому** вопросу про FASTQ со ссылкой на fasta-validator — там уже есть чужой ответ, ты дополняешь, не игра в одного актёра.

---

### Шаг 3 — только если через 2+ недели никто не ответил (опционально)

**Add answer** — коротко, без маркeting-тона:

```
What worked for me: two predict_sequence calls, same ontology, compare mean/max over the user segment (not N-padding). Still iterating on a tiny local UI for that — happy to share if useful once it's stable.

Disclosure: I'm building a small non-commercial compare tool for my own learning; not linking it here unless someone asks in comments.
```

Ссылку в комментарии — **только если спросят**. Так меньше похоже на astroturfing.

---

## Опционально (Biology SE — seqanalysis, у тебя больше кармы)

**Куда:** свой старый вопрос про FASTQ → **Add answer** (второй ответ, не дублируя первый)

**Текст:**

```
For a very quick FASTA format check in the browser (headers, alphabet, empty seqs) I use a small validator: https://seqanalysis.org/fasta-validator.html — no install. For FASTQ I still use FastQC/Fastp as in the answer above.

Disclosure: I maintain seqanalysis.org.
```
