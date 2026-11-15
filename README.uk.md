# POSEIDON — засвоєння даних ансамблевим методом (Python)

> 🇬🇧 English version: [README.md](README.md)

Інстальований Python-пакет (`poseidon_da`) для **засвоєння даних (data assimilation)** моделі
перенесення радіонуклідів **POSEIDON**. Реалізує **ансамблевий фільтр / згладжувач Калмана**:
коригує вектор стану POSEIDON за спостереженнями, обчислює аналіз у **просторі логарифмів
концентрацій**, записує оновлений стан у файли перезапуску `Initconc.dat`, після чого модель
продовжує прогноз до наступної дати.

Пакет надає стабільні **CLI** та **Python API**; деталі коваріації, індексації й розв'язувача
лишаються всередині пакета. Користувачеві потрібні лише набір даних, конфіг випадку та каталог
результатів.

---

## Зміст

- [Встановлення](#встановлення)
- [Швидкий старт (CLI)](#швидкий-старт-cli)
- [Python API](#python-api)
- [Конфігурація випадку](#конфігурація-випадку)
- [Опис методу](#опис-методу)
- [Відтворюваність](#відтворюваність)
- [Структура репозиторію](#структура-репозиторію)
- [Модулі пакета](#модулі-пакета)
- [Розробка та тести](#розробка-та-тести)

---

## Встановлення

Потрібен Python ≥ 3.11.

```bash
pip install .            # або: pip install -e .   (editable, для розробки)
```

Встановлюється пакет `poseidon_da` і команда `poseidon-da`. Пакет імпортується й запускається
з будь-якого каталогу, незалежно від вихідного дерева.

---

## Швидкий старт (CLI)

```bash
poseidon-da validate examples/black-sea-cs137.toml
poseidon-da analyze   examples/black-sea-cs137.toml -o results/window-01
poseidon-da run       examples/black-sea-cs137.toml -o results/run-01
poseidon-da diagnose  examples/black-sea-cs137.toml -o results/check-01
poseidon-da run-cycle examples/black-sea-cs137.toml -o results/cycle-01 --mode filter
```

- **validate** — перевірити конфіг і входи; вивести проблеми, ненульовий код при помилці.
- **analyze** — обчислити аналіз; записати `analysis.npz` і `result.json`.
- **run** — аналіз + файли перезапуску `Initconc.dat` + `out_<date>.xlsx` + `manifest.json`.
- **diagnose** — порівняти прогноз ансамблю зі спостереженнями (без засвоєння).
- **run-cycle** — послідовний цикл аналіз–прогноз по ланцюгу дат (`--mode noda|filter|smoother`; згладжувач використовує вікна ±`window_years`, точний порт оригінального run_LN_KSF3).

Прапорець `--no-perturb` — детермінований прогон (збурення спостережень = 0). Команди аналізу
моделі не потребують. Крок прогнозу (`run-cycle`) запускає `Poseidon4.exe`; у секції `[model]`
задайте `launcher = ["wine"]` на Linux/macOS (порожній на Windows) і `use_pty = true`, бо
POSEIDON — консольний Fortran-застосунок і headless-`wine` потребує псевдотермінала. Перевірено
наскрізно на macOS/arm64 (wine 11 staging): реальний 3-датний filter-цикл точно відтворив
архівні виходи прогнозу.

---

## Python API

```python
from poseidon_da import load_config, analyze, run

config = load_config("examples/black-sea-cs137.toml")

result = analyze(config, perturb=False)      # структурований результат, без запису файлів
print(result.target, result.log_states.shape)

run_result = run(config, "results/run-01")   # пише перезапуски, виходи, manifest
```

`analyze` повертає `AnalysisResult` (лог-стани, концентрації, геометричне середнє, індекси
спостережень, діагностика). Запис файлів — завжди окремий явний крок.

---

## Конфігурація випадку

Випадок — це TOML-файл; шляхи трактуються відносно каталогу файла.

```toml
[case]
schema_version = 1
case_id = "black-sea-cs137"

[dataset]
measurements_dir = "../measurements.exp2"
geometry         = "../config/Box_surfareas_with_centers.xlsx"
runs_root        = "../DAruns/01.RunKFS"

[analysis]
target_time       = "1988.5"
observation_times = ["1988.625", "1988.5"]   # один елемент → фільтр; кілька → згладжувач
gsd               = 1.3
bdepths           = [25, 100, 500, 2000]

[localization]
horizontal_m  = 100000.0
vertical_m    = 100.0
temporal_years = 1.0
exclude       = [1, 2, 3, 49, 50]

[model]
executable = "../bin/Poseidon4.exe"
launcher   = ["wine"]   # порожній на Windows
use_pty    = true       # консольному застосунку потрібен псевдотермінал
```

`target_time` має бути останнім у `observation_times`. Приклад: [examples/black-sea-cs137.toml](examples/black-sea-cs137.toml).

---

## Опис методу

**Область.** Чорне море розбите приблизно на 50 боксів по горизонталі та до 4 вертикальних
рівнів. Геометрія боксів (межі широти/довготи, рівні, центри мас) — у `config/*.xlsx`.

**Ансамбль.** 24 прогони POSEIDON (`NN.hydrolXdepY`) з різними сценаріями гідрології/осадження;
розкид ансамблю оцінює коваріацію похибок прогнозу.

**Простір логарифмів.** Аналіз виконується для натуральних логарифмів концентрацій
(логнормальні величини), що зберігає додатність оновлених концентрацій.

**Крок аналізу.** Для вікна аналізу:
1. зчитати й усереднити спостереження по боксах (`<date>.dat`);
2. зчитати стани ансамблю (`lastconc.xls`), побудувати розмітку стану й індекси спостережень;
3. оцінити ансамблеву коваріацію **C** і **локалізувати** її за горизонтальною/вертикальною
   (а для згладжувача — ще й часовою) відстанню; окремі бокси роблять незалежними;
4. розв'язати оновлення Калмана **у просторі спостережень**,
   `Xa = Xf + B Hᵀ (H B Hᵀ + R)⁻¹ (Y − H Xf)`, на кожного члена зі збуреними спостереженнями
   (стохастичний EnKF);
5. записати файли перезапуску та `out_<date>.xlsx`.

Одна дата у вікні — **фільтр**; кілька дат — **згладжувач** (аналіз на останню/цільову дату).

---

## Відтворюваність

Аналіз детермінований, окрім стохастичних збурень спостережень. Для повністю
детермінованого прогону використовуйте `--no-perturb` (CLI) або `perturb=False` (API), а для
відтворюваних збурених прогонів — `--seed` / `seed=`. Кожен прогін записує входи, параметри та
статус кроків у `manifest.json`.

Регресійний тест ([tests/test_regression.py](tests/test_regression.py)) звіряє розв'язувач із
збереженими еталонними масивами в `tests/reference/`; `tools/make_reference.py` формує
*candidate* для перегляду, не перезаписуючи baseline.

---

## Структура репозиторію

```text
poseidon-ensemble-da/
  pyproject.toml              # метадані пакета, залежності, console script
  src/poseidon_da/            # інстальований пакет
  examples/                   # приклад case.toml
  tests/                      # pytest + reference/ (еталонні масиви)
  tools/                      # make_reference.py
  config/                     # геометрія боксів (набір даних)
  measurements.exp2/          # файли спостережень (набір даних)
  DAruns/01.RunKFS/<date>/    # робочі каталоги POSEIDON (набір даних/архів)
  bin/Poseidon4.exe           # виконуваний файл моделі
```

Код інстальований і доступний лише для читання; набір даних (геометрія, вимірювання, сценарні
прогони) зберігається окремо й задається конфігом. Кожен прогон пише лише у свій каталог виходу.

---

## Модулі пакета

| Модуль | Призначення |
|---|---|
| `poseidon_da/api.py` | Публічний API: `load_config`, `analyze`, `run`. |
| `poseidon_da/cli.py`, `__main__.py` | CLI: `validate`, `analyze`, `run`, `diagnose`, `run-cycle`. |
| `poseidon_da/config.py` | Завантаження + валідація TOML у `Config`. |
| `poseidon_da/domain.py` | Типи-контракти (`StateLayout`, `AnalysisWindow`, `AnalysisResult`, `RunResult`). |
| `poseidon_da/workflow.py` | Збирає входи, запускає спільний аналіз, пише перезапуски/виходи/діагностику. |
| `poseidon_da/cycle.py` | Послідовний цикл NODA/filter (план → прогноз → діагностика) з походженням. |
| `poseidon_da/analysis.py` | `analysis_update` — оновлення Калмана у просторі спостережень (Холецький). |
| `poseidon_da/covariance.py` | Ансамблева коваріація + локалізація за відстанню (H/V/час). |
| `poseidon_da/state.py` | Розмітка стану (`build_layout`) + валідований `observation_indices`. |
| `poseidon_da/observations.py` | Геолокація + усереднення сирих вимірювань по боксах. |
| `poseidon_da/geometry.py` | Рівні боксів, глибини локалізації, центри боксів, валідація геометрії. |
| `poseidon_da/poseidon_io.py` | Читання/запис формату POSEIDON (із контрактом фізичного часу) + Excel. |
| `poseidon_da/runner.py` | `RunContext`, запуск `.exe`, копіювання файлів, перевірки прогону. |
| `poseidon_da/publish.py` | Транзакційний вихід: збірка в staging, атомарна публікація. |
| `poseidon_da/results.py` | Експорт (NPZ/JSON) + manifest. |
| `poseidon_da/paths.py` | Типові шляхи до даних (можна перевизначити конфігом). |

---

## Розробка та тести

```bash
pip install -e ".[test]"
pytest -q
```

Тести: регресія проти еталонів, алгебра ядра (замкнена форма Гаусса, `full = reduced`,
незв'язана координата незмінна), вхідні контракти (невалідний бокс/рівень, зіпсований
`lastconc.xls`, передумови runner'а), Excel I/O та конфіг/API/CLI на прикладі. Тести, що
потребують архіву, пропускаються без нього; усі прогони пишуть лише в ізольований `RunContext`.

`python tools/make_reference.py` формує *candidate*-масиви в `tests/reference/candidate/` для
перегляду — заморожений baseline не перезаписується.
