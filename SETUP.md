# Marine Autopilot HUD — Инструкция по запуску

## 1. Что нужно получить перед запуском

### Roboflow API Key
1. Зайди на https://app.roboflow.com
2. Войди в аккаунт (или создай)
3. Нажми на иконку профиля → **Roboflow API** → скопируй **Private API Key**

### Roboflow Workspace (slug)
1. В том же аккаунте посмотри URL — он выглядит так:
   `https://app.roboflow.com/MY-WORKSPACE-NAME/...`
2. Скопируй часть после `app.roboflow.com/` — это и есть workspace slug
   Пример: `john-doe-marine` или `konova1ove`

> Workflow ID уже зашит в код: `boat-autopilot-assistant-1774813329814`
> Его менять не нужно.

---

## 2. Установка зависимостей

```bash
# Убедись, что Python 3.9+ установлен
python3 --version

# Установи зависимости
pip install -r requirements.txt
```

Если pip ругается на права — используй:
```bash
pip install -r requirements.txt --user
```

Или через виртуальное окружение (рекомендуется):
```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## 3. Настройка .env файла

Скопируй шаблон:
```bash
cp .env.example .env
```

Открой `.env` и заполни:
```
ROBOFLOW_API_KEY=rf_aBcDeFgHiJkLmNoPqRsTuVwXyZ   ← твой Private API Key
ROBOFLOW_WORKSPACE=konova1ove                      ← твой workspace slug
CAMERA_INDEX=0                                     ← 0 = встроенная камера
                                                      1, 2... = USB-камеры
```

### Как найти индекс USB-камеры
```bash
# macOS / Linux — покажет доступные устройства
ls /dev/video*

# Или просто перебери: сначала попробуй 0, потом 1, 2 ...
# Если камера не открывается — в консоли будет:
# [ERROR] Cannot open camera. Check CAMERA_INDEX in .env
```

---

## 4. Запуск

```bash
cd /Users/polzovatel/Development/marine_autopilot

# Если используешь venv — сначала активируй:
source venv/bin/activate

python autopilot.py
```

Откроется окно **"Marine Autopilot HUD"** с видеопотоком и оверлеем.

---

## 5. Управление в окне

| Клавиша | Действие |
|---------|----------|
| `Q` | Выход |
| `+` / `=` | Увеличить горизонт проекции столкновения (+1 сек) |
| `-` | Уменьшить горизонт проекции (-1 сек) |
| `C` | Вывести текущие точки калибровки перспективы в консоль |

---

## 6. Калибровка перспективы (важно для точной скорости)

По умолчанию в `Config.PERSP_SRC` стоят примерные точки для камеры на высоте ~5 м.
Чтобы скорость считалась точнее — откалибруй под свою установку.

**Принцип:** выбери 4 точки на воде, для которых ты знаешь реальные расстояния
(например, углы сетки буёв или причала), и впиши их в `autopilot.py`:

```python
# В классе Config:
PERSP_SRC = np.float32([
    [пиксель_x, пиксель_y],   # точка 1 (ближний левый угол)
    [пиксель_x, пиксель_y],   # точка 2 (ближний правый)
    [пиксель_x, пиксель_y],   # точка 3 (дальний правый)
    [пиксель_x, пиксель_y],   # точка 4 (дальний левый)
])
PERSP_DST = np.float32([
    [0,   0],    # 0 м, 0 м
    [50,  0],    # 50 м вправо
    [50, 100],   # 50 м вправо, 100 м вперёд
    [0,  100],   # 0 м вправо, 100 м вперёд
])
```

Нажми `C` в окне — покажет текущие значения.

---

## 7. Переход на реальный CAN-bus

Когда будет физическое подключение к шине судна:

1. Установи python-can:
   ```bash
   pip install python-can
   ```
   (раскомментируй строку в `requirements.txt`)

2. В конце `autopilot.py` в функции `main()` замени одну строку:
   ```python
   # Было:
   canbus = MockCANBus()

   # Стало:
   canbus = RealCANBus(channel='can0', bustype='socketcan')
   ```

3. Дополни класс `RealCANBus` парсингом нужных PGN:
   - Heading → PGN 127250
   - Speed   → PGN 128259
   - RPM     → PGN 127488

Больше ничего менять не нужно — остальные модули работают через интерфейс `CANBusTelemetry`.

---

## 8. Типичные ошибки

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `[ERROR] Cannot open camera` | Неверный индекс | Попробуй `CAMERA_INDEX=1` или `2` |
| `[Roboflow] HTTP 401` | Неверный API Key | Проверь `.env`, пробелы/кавычки |
| `[Roboflow] HTTP 404` | Неверный workspace | Проверь slug в URL Roboflow |
| `[Roboflow] timeout` | Медленный интернет | Увеличь `timeout=5.0` в `RoboflowClient.infer()` |
| `ModuleNotFoundError: cv2` | OpenCV не установлен | `pip install opencv-python` |
