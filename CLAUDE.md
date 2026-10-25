# Robosumo RL

RL-політика для 3-кг робота сумо. Тренування в швидкому 2D-симі на GPU (PyTorch, тисячі середовищ паралельно),
експорт крихітної int8 MLP на Arduino Nano, далі цикл «залізо → логи → уточнення симу → перетренування».

Детальний план етапів: `PLAN.md`. Параметри робота, сенсорів, арени і суперника: `robot_spec.yaml`
(єдине джерело правди: жодних магічних чисел у коді, все читається звідти).

## Залізо (коротко)
- 3 кг, 200×200×130 мм, клин спереду, 2 колеса ⌀70 мм позаду, колія 150 мм, ЦМ ~2 см попереду осі.
- Мотори JGY-370 (черв'як, 1:40, самогальмування), 4S LiPo (14-16.8 В). Зчеплення слабше за мотори: повний газ = пробуксовка.
- Сенсори: 3× E18-D80NK (цифрові промені, 0° і ±15°), 4 line-сенсори в кутах квадрата 170 мм.
- МК: Arduino Nano (ATmega328P, без FPU, 2 КБ RAM). Політика тільки int8 fixed-point.
- Суперник: ідентичне залізо з класичним хардкод-FSM.

## Залізо (комп)
GTX 3070 8 GB, 32 GB RAM. Бюджет VRAM для тренування: до ~6 GB.

## Стек
- Python 3.12, `uv`, PyTorch (CUDA), numpy, PyYAML, pytest, tensorboard.
- Свій PPO у стилі CleanRL (один файл, читабельний). Без Gym-обгорток на CPU: середовище повністю векторизоване на GPU.
- Рендер матчів: matplotlib або pygame → mp4 (ffmpeg).

## Структура репо
```
robot_spec.yaml
sumo/
  config.py        # завантаження spec + randomization
  physics.py       # батчевий 2D rigid body, мотори, шини, контакти
  sensors.py       # ІЧ-промені, line-сенсори, шум/затримки/кросток
  opponents.py     # хардкод FSM (векторизовані)
  env.py           # SumoEnv: reset/step, obs/critic_obs, rewards, термінації
  render.py        # відео матчів
train/
  ppo.py           # PPO + asymmetric actor-critic
  league.py        # self-play пул суперників
  eval.py          # win-rate матриця, Elo, відео
export/
  quantize.py      # PyTorch → int8 ваги + скейли
  gen_header.py    # → firmware/policy_weights.h (PROGMEM)
  host_parity/     # C-реалізація інференсу для перевірки на ПК
firmware/          # Arduino-скетч: шари sensors → obs → policy → safety → motors
tests/
runs/              # чекпоінти + конфіг + git hash (в .gitignore)
```

## Правила роботи
- Один етап з `PLAN.md` = одна гілка. Не переходь до наступного, поки не виконано критерії «готово».
- Будь-яка зміна у фізиці або сенсорах → прогнати `pytest` і згенерувати відео матчу для ручної перевірки.
  RL знаходить і експлуатує баги фізики, тому тести на інваріанти обов'язкові.
- Спостереження актора містять ТІЛЬКИ те, що реально доступно на Nano. Привілейована інфа (позиції, швидкості,
  параметри рандомізації) йде лише критику.
- Фільтри, затримки, slew limit, deadband і квантування мають однакову реалізацію в симі і прошивці.
- Кожен чекпоінт зберігається з копією конфігу і git hash.
- Не збільшуй мережу без вимірювання часу інференсу на Nano.

## Середовище
- Репо на M: (NVMe). `UV_CACHE_DIR=M:\uv\cache`, `UV_PYTHON_INSTALL_DIR=M:\uv\python` (змінні користувача).
- torch з індексу `pytorch-cu128` (див. `pyproject.toml`). Системний msys64 `python` не використовувати.
- `arduino-cli` з ядром `arduino:avr` (FQBN Nano: `arduino:avr:nano`), `ffmpeg` у PATH.

## Команди (заповнювати по мірі появи)
- `uv run pytest` (`-s`, щоб побачити заміряну швидкість фізики)
- `uv run python -m sumo.render --demo` — відео-перевірка фізики в `runs/videos/`
- `uv run python -m train.ppo --config ...`
- `uv run python -m train.eval --ckpt ...`
- `uv run python -m export.gen_header --ckpt ...`
