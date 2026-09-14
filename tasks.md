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
- [x] **Download CSV** — экспорт таблицы Δ после Compare (и stats после single run)
- [x] **Push + live** — tissue + examples уже на `main` и на https://alphagenomio.onrender.com/ (проверено 2026-09-14). CSV выйдет на Render после merge в `main` (обычно auto-deploy).

### Что вставлять, если ты не биолог (проверка Live)

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
