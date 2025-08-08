## TODO: Интеграция Unity-клиента с сервером генерации Hunyuan3D-2.1
Не является проверенной инструкцией

Основано на инструкции из проекта Unity-клиента (OnGUI сцена) – см. репозиторий: [VR-Jobs/Hunyuan3D-2.1-Unity-XR-PC-Phone](https://github.com/VR-Jobs/Hunyuan3D-2.1-Unity-XR-PC-Phone)

### 0) Цель
- Поднять сервер генерации 3D и подключить Unity-клиент, который отправляет запросы на сервер и загружает/отображает результат (OBJ/GLB + текстуры).

### 1) Требования (Arch Linux / Desktop)
- NVIDIA драйвер и CUDA совместимы с используемым PyTorch (проверка: `nvidia-smi`).
- Установлен Python 3.10+ и `uv` (пакетный менеджер) – проект использует `uv`.
- Открыт порт, на котором стартует сервер (по умолчанию 8000) в firewall.
- Unity Hub и Unity Editor (LTS) установлены через AUR (`unityhub`) или официальный установщик. Рекомендуется 2021–2022 LTS.

### 2) Поднять сервер генерации
1. Установить зависимости:
   - Если нужно: `uv sync` (или используйте уже существующее окружение проекта).
2. Запуск сервера (WSGI + Granian) в корне репозитория:
   ```bash
   IMAGE_GEN_MODEL_ID=playgroundai/playground-v2.5-1024px-aesthetic \
   PYTHONPATH=frontend \
   uv run granian --interface wsgi --workers 1 --host 0.0.0.0 --port 8000 app:app
   ```
3. Проверить доступность: открыть `http://<SERVER_IP>:8000` в браузере.
4. Для удалённого сервера: убедиться, что порт 8000 открыт снаружи (security group / firewall).

Примечания:
- GPU/VRAM логируются при старте. Если VRAM не хватает, уменьшайте размер изображений/шагов или включайте offload.

### 3) Создать/подготовить Unity-проект
1. Создайте новый 3D-проект в Unity (или откройте существующий).
2. Установите пакет glTFast (для загрузки GLB/GLTF в рантайме):
   - Window → Package Manager → “+” → Add package from git URL
   - Вставьте URL: `https://github.com/atteneder/glTFast.git`
3. Установите Newtonsoft JSON:
   - Window → Package Manager → “+” → Add package by name
   - Введите: `com.unity.nuget.newtonsoft-json`
4. (Опционально) Скачайте тестовый пакет из Unity-репозитория (архив с OnGUI сценой) и распакуйте в `Assets/` вашего проекта. Откройте сцену `OnGUI`.

См. подробности у авторов Unity-клиента: [VR-Jobs/Hunyuan3D-2.1-Unity-XR-PC-Phone](https://github.com/VR-Jobs/Hunyuan3D-2.1-Unity-XR-PC-Phone)

### 4) Настройка Unity-сцены (OnGUI или своя UI)
1. Если используете их `OnGUI` сцену:
   - В инспекторе найдите компоненты, где задаются:
     - `Server IP` и `Port` – укажите `http://<SERVER_IP>:8000`
     - Локальная папка для скачивания результатов (например, `Application.persistentDataPath/models`)
2. Если делаете свою сцену/скрипт:
   - Добавьте MonoBehaviour, который:
     - формирует HTTP-запрос на сервер для запуска генерации;
     - периодически опрашивает статус;
     - по готовности скачивает результат (OBJ/GLB, MTL, текстуры);
     - грузит модель через glTFast (для GLB/GLTF) или через импорт OBJ.

### 5) Пример HTTP-вызовов из Unity (шаблон)
> Пример показывает общий подход. Конкретные URL/формат тела зависят от API сервера. Если предполагается совместимость с `flask_app.py` из Unity-репо, согласуйте эндпоинты (см. п. 6).

```csharp
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class HunyuanClient : MonoBehaviour
{
    [SerializeField] private string serverBase = "http://127.0.0.1:8000";

    public IEnumerator GenerateFromText(string prompt)
    {
        var url = serverBase + "/api/generate_textured_from_text"; // пример эндпоинта
        var payload = JsonUtility.ToJson(new Req { prompt = prompt });
        var req = new UnityWebRequest(url, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(payload);
        req.uploadHandler = new UploadHandlerRaw(bodyRaw);
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"HTTP Error: {req.error}");
            yield break;
        }

        // Ожидается job_id или прямая ссылка на результат
        var job = JsonUtility.FromJson<JobResp>(req.downloadHandler.text);
        // Далее – опрос статуса/скачивание
    }

    [System.Serializable]
    public class Req { public string prompt; }

    [System.Serializable]
    public class JobResp { public string job_id; public string download_url; }
}
```

### 6) Согласование API
В Unity-репозитории предлагают запустить их `flask_app.py` и работать с ожидаемыми ими ручками. Текущее решение - сервер (Flask + Granian) с другим UI и другими эндпоинтами. Варианты:

- Вариант A (быстро): добавить тонкий совместимый REST-слой (Blueprint) на сервере с минимальными ручками, которые ожидает Unity-клиент:
  - `POST /api/generate_from_text` → возвращает `{ job_id }`
  - `POST /api/generate_from_image` → возвращает `{ job_id }`
  - `GET /api/job/<job_id>/status` → `{ status: queued|running|done|error }`
  - `GET /api/job/<job_id>/result` → прямая ссылка/архив результатов (OBJ/GLB, MTL, текстуры)

- Вариант B (перенастройка клиента): если Unity-клиент позволяет задать конечные точки – укажите текущие ручки (например, заполняйте форму серверного UI через API и скачивайте ZIP по `download_*`).

Рекомендация: реализовать A – предсказуемые JSON-ручки под Unity, без изменения внутренних путей.

### 7) Тестовый прогон
1. Запустить сервер (см. п. 2) и убедиться, что GET `http://<SERVER_IP>:8000` открывается.
2. В Unity ввести `Server IP/Port`, нажать «Generate». Убедиться, что индикатор/лог в Unity показывает успешный запрос.
3. Проверить, что файлы модели появляются в целевой папке (или доступны по ссылке из ответа).
4. Для GLB/GLTF: загрузить через glTFast (компонент `GltfAsset` или рантайм API). Для OBJ: импорт через сторонний ридер или конвертация на стороне сервера в GLB.

### 8) Отладка
- Если Unity не подключается:
  - Проверить IP/порт, `curl http://<SERVER_IP>:8000` с машины, где работает Unity.
  - Проверить CORS (для WebGL билда) – при необходимости включить CORS на сервере.
- Если сервер генерирует, но Unity не показывает:
  - Проверить формат ответа API, пути к файлам, права на запись/чтение.
  - В логах сервера смотреть ошибки пайплайнов shape/paint.

### 9) Продакшн-заметки
- Реверс-прокси (nginx) перед сервером: SSL, лимиты, таймауты, кэш статики.
- Ограничение нагрузки на Python (concurrency/backpressure) – у Granian есть настройки.
- Хранение артефактов (модели/текстуры) – папка с периодической чисткой или объектное хранилище.

### Полезные ссылки
- Unity клиент и OnGUI сцена: [VR-Jobs/Hunyuan3D-2.1-Unity-XR-PC-Phone](https://github.com/VR-Jobs/Hunyuan3D-2.1-Unity-XR-PC-Phone)
- glTFast (импорт GLB/GLTF): `https://github.com/atteneder/glTFast`


