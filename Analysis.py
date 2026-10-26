from __future__ import annotations

import copy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

MAX_DISPLAY_WIDTH = 1000
MAX_DISPLAY_HEIGHT = 350
ROI_COLORS = ["#00ff66", "#ffb703", "#4cc9f0", "#ff5d8f", "#80ed99", "#d8b4ff"]


def analyze_roi(
    grayscale: np.ndarray, x: int, y: int, side: int
) -> tuple[float, float]:
    """Return the population standard deviation and mean intensity in an ROI."""
    if grayscale.ndim != 2:
        raise ValueError("The image must be a two-dimensional grayscale array.")
    if side < 1 or x < 0 or y < 0:
        raise ValueError("ROI coordinates must be non-negative and side length positive.")
    if x + side > grayscale.shape[1] or y + side > grayscale.shape[0]:
        raise ValueError("The ROI must fit entirely inside the image.")

    region = grayscale[y : y + side, x : x + side]
    return float(region.std(dtype=np.float64)), float(region.mean())


class ImageAnalysisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image ROI Analysis")
        self.root.minsize(640, 420)

        self.image: Image.Image | None = None
        self.grayscale: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.rois: list[dict[str, object]] = []
        self.active_roi_index = 0
        self.roi_x = 0
        self.roi_y = 0
        self.roi_side = 1
        self.drag_mode: str | None = None
        self.drag_offset_x = 0.0
        self.drag_offset_y = 0.0
        self.undo_stack: list[list[dict[str, object]]] = []
        self.redo_stack: list[list[dict[str, object]]] = []
        self._mouse_move_history = None

        controls = ttk.Frame(root, padding=(6, 3))
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Open image...", command=self.open_image).pack(
            side=tk.LEFT
        )
        ttk.Button(controls, text="Add square", command=self.add_square).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(controls, text="↶", command=self.undo, width=2).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(controls, text="↷", command=self.redo, width=2).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(
            controls, text="Copy table", command=self.copy_table_values
        ).pack(side=tk.LEFT, padx=(0, 0))

        self.size_label = ttk.Label(controls, text="Square size: ")
        self.size_label.pack(side=tk.LEFT, padx=(8, 4))
        self.size_entry = ttk.Entry(controls, width=5)
        self.size_entry.pack(side=tk.LEFT)
        # ttk.Button(controls, text="Apply", command=self._apply_size_entry).pack(
        #     side=tk.LEFT, padx=(6, 0)
        # )
        self.size_control = tk.Scale(
            controls,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            showvalue=True,
            length=220,
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
        self.canvas.pack(padx=6, pady=(0, 3))
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.focus_set()
        self.root.bind_all("<Left>", self._on_arrow_key)
        self.root.bind_all("<Right>", self._on_arrow_key)
        self.root.bind_all("<Up>", self._on_arrow_key)
        self.root.bind_all("<Down>", self._on_arrow_key)
        self.size_entry.bind("<Return>", lambda event: self._apply_size_entry())

        # results = ttk.Frame(root, padding=(8, 0, 8, 8))
        # results.pack(fill=tk.X)
        # self.standard_deviation = ttk.Label(
        #     results, text="Standard deviation: --"
        # )
        # self.standard_deviation.pack(side=tk.LEFT, padx=(0, 24))
        # self.mean_grayscale = ttk.Label(
        #     results, text="Mean grayscale: --"
        # )
        # self.mean_grayscale.pack(side=tk.LEFT)

        self.roi_table = ttk.LabelFrame(root, text="ROI Table", padding=2)
        self.roi_table.pack(fill=tk.X, padx=6, pady=(0, 3))

        # self.roi_overview = ttk.Label(
        #     root,
        #     text="ROI stats: --",
        #     padding=(8, 0, 8, 8),
        # )
        # self.roi_overview.pack(anchor=tk.W)
        # self.help_text = ttk.Label(
        #     root,
        #     text="Click a square to select it, drag inside to move it, or drag the lower-right corner to resize it. Arrow keys move the selected ROI by 1 pixel.",
        #     padding=(8, 0, 8, 8),
        # )
        # self.help_text.pack(anchor=tk.W)

    def _active_roi(self) -> dict[str, object] | None:
        if not self.rois:
            return None
        return self.rois[self.active_roi_index]

    def _sync_active_roi_state(self) -> None:
        active = self._active_roi()
        if active is None:
            return
        self.roi_x = int(active["x"])
        self.roi_y = int(active["y"])
        self.roi_side = int(active["side"])
        self.size_entry.delete(0, tk.END)
        self.size_entry.insert(0, str(self.roi_side))

    def _roi_under_pointer(self, x: float, y: float) -> int | None:
        for index in range(len(self.rois) - 1, -1, -1):
            roi = self.rois[index]
            left = float(roi["x"]) * self.display_scale
            top = float(roi["y"]) * self.display_scale
            right = left + float(roi["side"]) * self.display_scale
            bottom = top + float(roi["side"]) * self.display_scale
            if left <= x <= right and top <= y <= bottom:
                return index
        return None

    def _snapshot_rois(self) -> list[dict[str, object]]:
        return copy.deepcopy(self.rois)

    def _record_history(self, before_state: list[dict[str, object]]) -> None:
        snapshot = copy.deepcopy(before_state)
        if self.undo_stack and self.undo_stack[-1] == snapshot:
            return
        self.undo_stack.append(snapshot)
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _push_history(self) -> None:
        self._record_history(self._snapshot_rois())

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot_rois())
        self.rois = copy.deepcopy(self.undo_stack.pop())
        self.active_roi_index = min(self.active_roi_index, len(self.rois) - 1) if self.rois else 0
        self._sync_active_roi_state()
        self._render()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot_rois())
        self.rois = copy.deepcopy(self.redo_stack.pop())
        self.active_roi_index = min(self.active_roi_index, len(self.rois) - 1) if self.rois else 0
        self._sync_active_roi_state()
        self._render()

    def _select_roi(self, index: int) -> None:
        if not self.rois or index >= len(self.rois):
            return
        self.active_roi_index = index
        self._sync_active_roi_state()
        self._render()

    def _populate_roi_table(self) -> None:
        for child in self.roi_table.winfo_children():
            child.destroy()

        if not self.rois:
            return

        for index, roi in enumerate(self.rois):
            row = index // 3
            column = index % 3
            card = ttk.Frame(self.roi_table, padding=2, relief=tk.SOLID, borderwidth=1)
            card.grid(row=row, column=column, padx=2, pady=1, sticky="nsew")
            if index == self.active_roi_index:
                card.configure(borderwidth=2)

            deviation, mean = analyze_roi(
                self.grayscale,
                int(roi["x"]),
                int(roi["y"]),
                int(roi["side"]),
            )
            stats = ttk.Label(
                card,
                text=(
                    f"ROI {index + 1}  |  Size: {roi['side']} px\n"
                    f"SD: {deviation:.2f}  |  Mean: {mean:.1f}"
                ),
                justify=tk.LEFT,
            )
            stats.pack(fill=tk.X)

        for column_index in range(3):
            self.roi_table.grid_columnconfigure(
                column_index, weight=1, uniform="roi_columns"
            )   

    def _reset_rois(self) -> None:
        if self.image is None:
            return
        side = max(1, int(min(self.image.width, self.image.height) * 0.25))
        x = (self.image.width - side) // 2
        y = (self.image.height - side) // 2
        self.rois = [
            {
                "x": x,
                "y": y,
                "side": side,
                "color": ROI_COLORS[0],
                "label": "1",
            }
        ]
        self.active_roi_index = 0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._sync_active_roi_state()
        self.size_control.configure(to=min(self.image.width, self.image.height), state=tk.NORMAL)
        self.size_control.set(self.roi_side)

    def add_square(self) -> None:
        if self.image is None or self.grayscale is None:
            return

        base = self._active_roi()
        if base is None:
            self._reset_rois()
            base = self._active_roi()

        side = max(1, int(base["side"]))
        x = int(base["x"]) + 24
        y = int(base["y"]) + 24
        color_index = len(self.rois) % len(ROI_COLORS)
        new_x = min(max(x, 0), self.image.width - side)
        new_y = min(max(y, 0), self.image.height - side)
        self._record_history(self._snapshot_rois())
        self.rois.append(
            {
                "x": new_x,
                "y": new_y,
                "side": side,
                "color": ROI_COLORS[color_index],
                "label": str(len(self.rois) + 1),
            }
        )
        self.active_roi_index = len(self.rois) - 1
        self._sync_active_roi_state()
        self.size_control.set(self.roi_side)
        self._render()

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
            MAX_DISPLAY_WIDTH / image.width,
            MAX_DISPLAY_HEIGHT / image.height,
            1.0,
        )
        self._reset_rois()
        self._render()

    def _apply_size_entry(self) -> None:
        if self.image is None:
            return
        active = self._active_roi()
        if active is None:
            return
        try:
            value = int(self.size_entry.get())
        except ValueError:
            self.size_entry.delete(0, tk.END)
            self.size_entry.insert(0, str(self.roi_side))
            return

        max_size = min(self.image.width, self.image.height)
        if value < 1:
            value = 1
        elif value > max_size:
            value = max_size

        if int(active["side"]) == value:
            return

        self._record_history(self._snapshot_rois())
        active["side"] = value
        self._keep_roi_in_image()
        self._render()

    def _on_size_change(self, value: str) -> None:
        if self.image is None:
            return
        active = self._active_roi()
        if active is None:
            return
        new_size = max(1, int(float(value)))
        if new_size == int(active["side"]):
            return
        before = self._snapshot_rois()
        active["side"] = new_size
        if before != self.rois:
            self._record_history(before)
        self._keep_roi_in_image()
        self._render()

    def _on_arrow_key(self, event: tk.Event) -> str:
        if self.image is None:
            return "break"
        active = self._active_roi()
        if active is None:
            return "break"

        before = self._snapshot_rois()
        x = 0
        y = 0
        if event.keysym == "Left":
            x = -2
        elif event.keysym == "Right":
            x = 2
        elif event.keysym == "Up":
            y = -2
        elif event.keysym == "Down":
            y = 2
        if event.keysym == "Shift":
            x *= 2
            y *= 2

        active["x"] = int(active["x"]) + x
        active["y"] = int(active["y"]) + y

        if before != self.rois:
            self._record_history(before)
        self._keep_roi_in_image()
        self._render()
        return "break"

    def _on_mouse_down(self, event: tk.Event[tk.Canvas]) -> None:
        if self.image is None:
            return

        selected_index = self._roi_under_pointer(event.x, event.y)
        if selected_index is not None:
            self.active_roi_index = selected_index
            self._sync_active_roi_state()

        active = self._active_roi()
        if active is None:
            return

        self._mouse_move_history = copy.deepcopy(self.rois)
        side_on_canvas = int(active["side"]) * self.display_scale
        left = int(active["x"]) * self.display_scale
        top = int(active["y"]) * self.display_scale
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
                active["x"] = round(event.x / self.display_scale - int(active["side"]) / 2)
                active["y"] = round(event.y / self.display_scale - int(active["side"]) / 2)
                self._keep_roi_in_image()
            self.drag_mode = "move"
            self.drag_offset_x = event.x - int(active["x"]) * self.display_scale
            self.drag_offset_y = event.y - int(active["y"]) * self.display_scale

        self._render()

    def _on_mouse_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self.image is None or self.drag_mode is None:
            return

        active = self._active_roi()
        if active is None:
            return

        if self.drag_mode == "resize":
            available = min(
                self.image.width - int(active["x"]), self.image.height - int(active["y"])
            )
            width = event.x / self.display_scale - int(active["x"])
            height = event.y / self.display_scale - int(active["y"])
            active["side"] = max(1, min(available, round(max(width, height))))
            self.size_control.set(int(active["side"]))
        else:
            active["x"] = round((event.x - self.drag_offset_x) / self.display_scale)
            active["y"] = round((event.y - self.drag_offset_y) / self.display_scale)
            self._keep_roi_in_image()

        self._render()

    def _on_mouse_up(self, event: tk.Event[tk.Canvas]) -> None:
        if self._mouse_move_history is not None:
            if self._mouse_move_history != self.rois:
                self._record_history(self._mouse_move_history)
            self._mouse_move_history = None
        self.drag_mode = None
        self._render()

    def _keep_roi_in_image(self) -> None:
        if self.image is None:
            return
        active = self._active_roi()
        if active is None:
            return
        active["x"] = min(max(int(active["x"]), 0), self.image.width - int(active["side"]))
        active["y"] = min(max(int(active["y"]), 0), self.image.height - int(active["side"]))
        self._sync_active_roi_state()

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

        for roi in self.rois:
            left = float(roi["x"]) * self.display_scale
            top = float(roi["y"]) * self.display_scale
            side_on_canvas = float(roi["side"]) * self.display_scale
            outline = str(roi["color"])
            width = 3 if roi is self._active_roi() else 2
            self.canvas.create_rectangle(
                left,
                top,
                left + side_on_canvas,
                top + side_on_canvas,
                outline=outline,
                width=width,
            )
            self.canvas.create_rectangle(
                left + side_on_canvas - 5,
                top + side_on_canvas - 5,
                left + side_on_canvas + 5,
                top + side_on_canvas + 5,
                fill=outline,
                outline=outline,
            )
            self.canvas.create_text(
                left + 7,
                top + 6,
                text=str(roi["label"]),
                anchor=tk.NW,
                fill="#ffffff",
                font=("TkDefaultFont", 9, "bold"),
            )

        active = self._active_roi()
        if active is not None:
            self.roi_x = int(active["x"])
            self.roi_y = int(active["y"])
            self.roi_side = int(active["side"])
            # self.size_label.configure(text=f"Square size: {self.roi_side} px")
            self.size_control.set(self.roi_side)
            self.size_entry.delete(0, tk.END)
            self.size_entry.insert(0, str(self.roi_side))
            deviation, mean = analyze_roi(
                self.grayscale,
                self.roi_x,
                self.roi_y,
                self.roi_side,
            )
            # self.standard_deviation.configure(
            #     text=f"Standard deviation: {deviation:.3f} (8-bit grayscale)"
            # )
            # self.mean_grayscale.configure(
            #     text=f"Mean grayscale: {mean:.1f} (0-255)"
            # )

        # lines: list[str] = []
        # for index, roi in enumerate(self.rois, start=1):
        #     deviation, mean = analyze_roi(
        #         self.grayscale,
        #         int(roi["x"]),
        #         int(roi["y"]),
        #         int(roi["side"]),
        #     )
        #     lines.append(f"ROI {index}: SD {deviation:.3f} | Mean {mean:.1f}")
        # self.roi_overview.configure(text="  ".join(lines))
        self._populate_roi_table()

    def copy_table_values(self) -> None:
        if self.grayscale is None or not self.rois:
            messagebox.showinfo("No ROI data", "Open an image to copy ROI values.", parent=self.root)
            return

        rows = ["ROI\tSize (px)\tStandard deviation\tMean grayscale"]
        for index, roi in enumerate(self.rois, start=1):
            deviation, mean = analyze_roi(
                self.grayscale,
                int(roi["x"]),
                int(roi["y"]),
                int(roi["side"]),
            )
            rows.append(
                f"{index}\t{roi['side']}\t{deviation:.2f}\t{mean:.1f}"
            )

        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(rows))
        self.root.update()
        messagebox.showinfo("Copied", "ROI table values copied to the clipboard.", parent=self.root)


def main() -> None:
    root = tk.Tk()
    ImageAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()