import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pandasai.responses import ResponseParser


class CustomResponseParser(ResponseParser):
    def format_plot(self, result: dict) -> Any:
        super().format_plot(result)
        filename = str(uuid.uuid4()).replace("-", "")

        temp_image_path = Path(f"{tempfile.tempdir}/streamlit/{filename}.png")
        temp_image_path.parent.mkdir(parents=True, exist_ok=True)

        original_path = Path("temp_chart.png")
        shutil.copy(original_path, temp_image_path)
        print("image created: ", str(temp_image_path))
        return {"type": "plot", "value": str(temp_image_path)}
