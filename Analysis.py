from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

disp_width = 1000
disp_height = 700


def analyze_roi(
    grayscale: np.ndarray, x: int, y: int, side: int
) -> tuple[float, int]:
    """Return the population standard deviation and minimum intensity in an ROI."""
    if grayscale.ndim != 2:
        raise ValueError("The image must be a two-dimensional grayscale array.")
    if side < 1 or x < 0 or y < 0:
        raise ValueError("ROI coordinates must be non-negative and side length positive.")
    if x + side > grayscale.shape[1] or y + side > grayscale.shape[0]:
        raise ValueError("The ROI must fit entirely inside the image.")

    region = grayscale[y : y + side, x : x + side]
    return float(region.std(dtype=np.float64)), int(region.min())


class ImageAnalysisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image ROI Analysis")
        self.root.minsize(640, 420)

        self.image: Image.Image | None = None
        self.grayscale: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.roi_x = 0
        self.roi_y = 0
        self.roi_side = 1
        self.drag_mode: str | None = None
        self.drag_offset_x = 0.0
        self.drag_offset_y = 0.0

        controls = ttk.Frame(root, padding=8)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Open image...", command=self.open_image).pack(
            side=tk.LEFT
        )
        self.size_label = ttk.Label(controls, text="Square size: --")
        self.size_label.pack(side=tk.LEFT, padx=(16, 4))
        self.size_control = tk.Scale(
            controls,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            showvalue=True,
            length=260,
            command=self._on_size_change,
            state=tk.DISABLED,
        )
        self.size_control.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.canvas = tk.Canvas(
            root,
            width=800,
            height=500,
            background="#333333",
            highlightthickness=0,
        )
        self.canvas.pack(padx=8, pady=(0, 8))
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        results = ttk.Frame(root, padding=(8, 0, 8, 8))
        results.pack(fill=tk.X)
        self.standard_deviation = ttk.Label(
            results, text="Standard deviation: --"
        )
        self.standard_deviation.pack(side=tk.LEFT, padx=(0, 24))
        self.blackpoint = ttk.Label(
            results, text="Blackpoint (minimum grayscale, 0-255): --"
        )
        self.blackpoint.pack(side=tk.LEFT)
        self.help_text = ttk.Label(
            root,
            text="Drag inside the square to move it; drag its lower-right corner "
            "to resize. Use the slider for precise sizing.",
            padding=(8, 0, 8, 8),
        )
        self.help_text.pack(anchor=tk.W)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).copy()
        except (OSError, ValueError) as error:
            messagebox.showerror("Unable to open image", str(error), parent=self.root)
            return

        self.image = image
        self.grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        self.display_scale = min(
            disp_width / image.width,
            disp_height / image.height,
            1.0,
        )
        self.roi_side = max(1, int(min(image.width, image.height) * 0.25))
        self.roi_x = (image.width - self.roi_side) // 2
        self.roi_y = (image.height - self.roi_side) // 2

        self.size_control.configure(to=min(image.width, image.height), state=tk.NORMAL)
        self.size_control.set(self.roi_side)
        self._render()

    def _on_size_change(self, value: str) -> None:
        if self.image is None:
            return
        self.roi_side = max(1, int(float(value)))
        self._keep_roi_in_image()
        self._render()

    def _on_mouse_down(self, event: tk.Event[tk.Canvas]) -> None:
        if self.image is None:
            return

        side_on_canvas = self.roi_side * self.display_scale
        left = self.roi_x * self.display_scale
        top = self.roi_y * self.display_scale
        near_resize_corner = (
            abs(event.x - (left + side_on_canvas)) <= 12
            and abs(event.y - (top + side_on_canvas)) <= 12
        )

        if near_resize_corner:
            self.drag_mode = "resize"
        else:
            if not (
                left <= event.x <= left + side_on_canvas
                and top <= event.y <= top + side_on_canvas
            ):
                self.roi_x = round(event.x / self.display_scale - self.roi_side / 2)
                self.roi_y = round(event.y / self.display_scale - self.roi_side / 2)
                self._keep_roi_in_image()
            self.drag_mode = "move"
            self.drag_offset_x = event.x - self.roi_x * self.display_scale
            self.drag_offset_y = event.y - self.roi_y * self.display_scale

        self._render()

    def _on_mouse_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self.image is None or self.drag_mode is None:
            return

        if self.drag_mode == "resize":
            available = min(
                self.image.width - self.roi_x, self.image.height - self.roi_y
            )
            width = event.x / self.display_scale - self.roi_x
            height = event.y / self.display_scale - self.roi_y
            self.roi_side = max(1, min(available, round(max(width, height))))
            self.size_control.set(self.roi_side)
        else:
            self.roi_x = round(
                (event.x - self.drag_offset_x) / self.display_scale
            )
            self.roi_y = round(
                (event.y - self.drag_offset_y) / self.display_scale
            )
            self._keep_roi_in_image()

        self._render()

    def _on_mouse_up(self, event: tk.Event[tk.Canvas]) -> None:
        self._on_mouse_drag(event)
        self.drag_mode = None

    def _keep_roi_in_image(self) -> None:
        if self.image is None:
            return
        self.roi_x = min(max(self.roi_x, 0), self.image.width - self.roi_side)
        self.roi_y = min(max(self.roi_y, 0), self.image.height - self.roi_side)

    def _render(self) -> None:
        if self.image is None or self.grayscale is None:
            return

        display_width = max(1, round(self.image.width * self.display_scale))
        display_height = max(1, round(self.image.height * self.display_scale))
        display_image = self.image.convert("RGB").resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(display_image)
        self.canvas.configure(width=display_width, height=display_height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        left = self.roi_x * self.display_scale
        top = self.roi_y * self.display_scale
        side_on_canvas = self.roi_side * self.display_scale
        self.canvas.create_rectangle(
            left,
            top,
            left + side_on_canvas,
            top + side_on_canvas,
            outline="#00ff66",
            width=2,
        )
        self.canvas.create_rectangle(
            left + side_on_canvas - 5,
            top + side_on_canvas - 5,
            left + side_on_canvas + 5,
            top + side_on_canvas + 5,
            fill="#00ff66",
            outline="#00ff66",
        )

        deviation, blackpoint = analyze_roi(
            self.grayscale, self.roi_x, self.roi_y, self.roi_side
        )
        self.size_label.configure(text=f"Square size: {self.roi_side} px")
        self.standard_deviation.configure(
            text=f"Standard deviation: {deviation:.3f} (8-bit grayscale)"
        )
        self.blackpoint.configure(
            text=f"Blackpoint (minimum grayscale, 0-255): {blackpoint}"
        )


def main() -> None:
    root = tk.Tk()
    ImageAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()