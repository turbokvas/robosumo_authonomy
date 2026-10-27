# Robosumo RL

RL-політика для 3-кг робота сумо. M0–M4 натреновано в швидкому 2D-симі на GPU (PyTorch, тисячі середовищ паралельно);
з M5 тренування йде в 3D (MuJoCo Warp, геометрія з CAD: підчеп, відрив коліс, ковзання шин). Далі експорт політики на МК
(ESP32 мінімум, STM32 бажано) і цикл «залізо → логи → уточнення симу → перетренування». Заліза поки немає.
Де ми зараз і що далі: `STATUS.md`. Журнал знахідок і відкриті питання: `PROGRESS.md`.

Детальний план етапів: `PLAN.md`. Параметри робота, сенсорів, арени і суперника: `robot_spec.yaml`
(єдине джерело правди: жодних магічних чисел у коді, все читається звідти).

## Залізо (коротко)
- 3 кг, 190×199×99 мм (CAD: `sim3d/assets/robot_mdl.step`), клин спереду, 2 колеса ⌀70 мм позаду, колія 149 мм, ЦМ ~2 см попереду осі.
- Мотори JGY-370 (черв'як, 1:40, самогальмування), 4S LiPo (14-16.8 В). Зчеплення слабше за мотори: повний газ = пробуксовка.
- Сенсори: 3× E18-D80NK (цифрові промені, 0° і ±15°), 4 line-сенсори в кутах квадрата 170 мм.
- МК: ще не обрано — мінімум ESP32, бажано STM32 (обидва з FPU). Nano (торішній робот) НЕ ціль, під нього не обмежуємось.
  Пріоритет — якість політики; розмір мережі/частоту обмежимо заміром на реальному МК, коли він буде.
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
  interface.py     # ЗАМОРОЖЕНИЙ інтерфейс політики (v2): склад obs/action/critic, версія чекпоінтів
  firmware.py      # шар прошивки: slew limit, компенсація мертвої зони, пам'ять сенсорів (той самий код піде в C)
  config.py        # завантаження spec + randomization, оверрайди spec
  physics.py       # батчевий 2D rigid body, мотори, шини, контакти
  sensors.py       # ІЧ-промені, line-сенсори, шум/затримки/кросток
  opponents.py     # хардкод FSM (векторизовані)
  env.py           # SumoEnv: reset/step, obs/critic_obs, rewards, термінації
  render.py        # відео матчів
  match.py         # FSM проти FSM: win-rate, матриця, відео
sim3d/             # 3D: build_assets (STEP→меші/бокси), model (MJCF), env3d (SumoEnv3D, той самий інтерфейс), render3d
train/
  ppo.py           # PPO + asymmetric actor-critic
  league.py        # self-play пул суперників
  eval.py          # win-rate матриця, Elo, відео
export/
  export.py        # PyTorch → C-хедер з вагами (float32; int8 опційно)
  host_parity/     # C-реалізація інференсу для перевірки на ПК
firmware/          # ESP32/STM32 (PlatformIO): шари sensors → obs → policy → safety → motors
tests/
runs/              # чекпоінти + конфіг + git hash (в .gitignore)
```

## Правила роботи
- Один етап з `PLAN.md` = одна гілка. Не переходь до наступного, поки не виконано критерії «готово».
- Будь-яка зміна у фізиці або сенсорах → прогнати `pytest` і згенерувати відео матчу для ручної перевірки.
  RL знаходить і експлуатує баги фізики, тому тести на інваріанти обов'язкові.
- Спостереження актора містять ТІЛЬКИ те, що реально доступно на МК робота. Привілейована інфа (позиції, швидкості,
  параметри рандомізації) йде лише критику.
- Фільтри, затримки, slew limit, deadband і квантування мають однакову реалізацію в симі і прошивці.
- Кожен чекпоінт зберігається з копією конфігу і git hash.
- Інтерфейс політики заморожений (`sumo/interface.py`, зараз v2: кадри 0–320 мс + пам'ять прошивки). Склад, порядок, масштаби входів, семантику дії,
  frame stack і розмір мережі (64-64) не міняти без запиту користувача; зміна = нова `INTERFACE_VERSION`, запис у PLAN.md,
  старі чекпоінти несумісні. Нові сенсори лише через резервні слоти. Детально: `INTERFACE_AND_M5.md`.
- Не збільшуй мережу без заміру часу інференсу на МК (M7, бюджет < половини циклу) і підтвердження користувача.
- Не видаляй старі чекпоінти і результати eval без прямого прохання. Блокери й відкриті питання — в кінець `PROGRESS.md`.
- Поки йде тренування, нічого іншого на GPU не запускати (smoke-тести, бенчі, великі eval): 8 GB на двох не вистачає,
  і комп власника задихається. Малі рендери (4 середовища) — можна.
- Цифри, які звітуєш власнику, перевіряй тим самим кодом, що й тренер (`train.eval`); нічия в коді = −1 (`sumo.env.DRAW`).
- Фізику 3D не міняти посеред етапу без погодження з власником: кожна зміна скидає прогрес тренування.

## Середовище
- Репо на M: (NVMe). `UV_CACHE_DIR=M:\uv\cache`, `UV_PYTHON_INSTALL_DIR=M:\uv\python` (змінні користувача).
- torch з індексу `pytorch-cu128` (див. `pyproject.toml`). Системний msys64 `python` не використовувати.
- `ffmpeg` у PATH. `arduino-cli` встановлено (ядра avr, esp32), але прошивку плануємо через PlatformIO під ESP32/STM32.

## Команди (заповнювати по мірі появи)
- `uv run pytest` (`-s`, щоб побачити заміряну швидкість фізики)
- `uv run python -m sumo.render --demo` — відео-перевірка фізики в `runs/videos/`
- `uv run python -m sumo.match --a aggressive --b random_walker --n 4096 [--nominal] [--video 6]` — FSM проти FSM
- `uv run python -m sumo.match --matrix --n 1024 --nominal` — матриця win-rate усіх варіантів FSM
- `uv run python -m train.ppo --config configs/m3.yaml` — тренування (tensorboard: `uv run tensorboard --logdir runs`)
- `uv run python -m train.eval --ckpt runs/<run>/ckpt_latest.pt [--video 2]` — win-rate проти всіх суперників
- `uv run python scripts/bench_env.py --n 4096` — швидкість середовища
- `uv run python -m sim3d.build_assets` — перегенерувати 3D-ассети з STEP (після змін у Fusion)
- `uv run python -m train.eval --ckpt ... --sim 3d` — оцінка в 3D; `uv run python -m sim3d.render3d --ckpt ... --out x.mp4` — 3D-відео
- 3D-тренування: `env.sim: 3d` у конфігу (див. `configs/m5.yaml`); `spec_overrides:` у конфігу змінює spec для прогону
- `uv run python -m train.ppo --resume runs/<run>/ckpt_latest.pt [--total-steps N]` — продовжити після обриву
- `uv run python -m train.eval --ckpt ... --extreme configs/eval_extreme.yaml --sim 3d` — номінал vs екстремальне залізо
- `uv run python -m train.eval --ckpt ... --regression configs/regression.yaml --baseline results/m4/ckpt_02000M.pt`
- `uv run python -m export.export --ckpt ...`
