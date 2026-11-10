######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.10
# filenaem: visualize.py
# function: draw the result of a segmentation model on an image.
#
# the module holds the drawing of the predictions:
#   1. the mask of a class is blended into the image, e.g. the hair is drawn
#      with 70% of the colour of the class and 30% of the original pixels.
#   2. a legend of the drawn classes is written on the picture.
#   3. the label (or the original image when there is no label) and the
#      prediction are put next to each other in a single picture.
######################################################
import os
import logging

import cv2
import numpy as np

_logger = logging.getLogger("visualize")

# the colour of every class, in BGR (the order opencv uses).  the index of a
# colour is the class id.
DEFAULT_PALETTE = [
    (0, 0, 0),        # 0: background (black)
    (0, 0, 255),      # 1: hair (red)
    (0, 255, 0),      # 2: (green)
    (255, 0, 0),      # 3: (blue)
    (0, 255, 255),    # 4: (yellow)
    (255, 0, 255),    # 5: (magenta)
    (255, 255, 0),    # 6: (cyan)
    (0, 128, 255),    # 7: (orange)
]


def get_palette(num_classes, palette=None):
    """Return a palette of ``num_classes`` BGR colours.

    The given palette is used as long as it is long enough; the missing
    colours are taken from the default one.
    """
    colors = list(palette) if palette else list(DEFAULT_PALETTE)
    if not colors:
        colors = list(DEFAULT_PALETTE)

    if len(colors) < num_classes:
        # repeat the default palette until there are enough colours
        extra = list(DEFAULT_PALETTE)
        while len(colors) < num_classes:
            colors.append(extra[(len(colors) - len(DEFAULT_PALETTE)) % len(extra)])
    return colors[:num_classes]


def parse_palette(palette):
    """Turn a palette of the config into a list of BGR tuples.

    A palette written in the yaml is a list of lists (e.g.
    ``[[0, 0, 0], [0, 0, 255]]``); a palette given on the command line is cut
    into a flat list of strings (e.g. ``['0', '0', '0', '0', '0', '255']``),
    because the cli overrides are split on the commas.

    Returns:
        A list of ``(B, G, R)`` tuples.
    """
    if not palette:
        return []

    # a flat string: "0,0,0;0,0,255"
    if isinstance(palette, str):
        palette = palette.replace(";", ",").split(",")

    palette = [c for c in palette if str(c).strip()]

    colors = []
    current = []
    for value in palette:
        if isinstance(value, (list, tuple)):
            # a nested list: one colour
            colors.append(tuple(int(v) for v in value))
            continue

        # a flat list: the channels of the colours, three by three
        text = str(value).strip()
        if not text:
            continue
        try:
            current.append(int(text))
        except ValueError:
            # the value holds several channels at once, e.g. "0,0,255"
            for part in text.replace(";", ",").split(","):
                part = part.strip()
                if part:
                    current.append(int(part))

        if len(current) >= 3:
            colors.append(tuple(current[:3]))
            current = current[3:]

    if current:
        raise ValueError(
            f"the palette must hold three channels per colour, got {palette}."
        )
    return colors


def blend_mask(image, mask, color, ratio=0.7):
    """Blend the pixels of a mask into an image.

    The pixels of the mask are replaced by ``ratio`` of the colour and
    ``1 - ratio`` of the original pixels.

    Args:
        image: the original image, a ``uint8`` BGR array.
        mask: the pixels to draw, a boolean array of the shape of the image.
        color: the BGR colour of the mask.
        ratio: the share of the colour in the result (0.7 -> 70% colour).

    Returns:
        A new image with the mask blended into it.
    """
    if ratio <= 0:
        return image.copy()
    if ratio >= 1:
        out = image.copy()
        out[mask] = color
        return out

    out = image.astype(np.float32)
    color = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - ratio) + color * ratio
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_prediction(image, pred, classes=None, palette=None, ratio=0.7,
                       class_ids=None, dim_background=0.0):
    """Draw a predicted mask on an image.

    Args:
        image: the original image, a ``uint8`` BGR array.
        pred: the predicted class ids, an array of the shape of the image.
        classes: the names of the classes (used for the legend).
        palette: the BGR colour of every class.
        ratio: the share of the colour of the mask (0.7 -> 70% colour).
        class_ids: the classes that are drawn; every class but the background
            is drawn when it is None.
        dim_background: how much the pixels that are not drawn are dimmed
            (0.0 keeps them untouched).

    Returns:
        A new image with the prediction drawn on it.
    """
    colors = get_palette(len(classes) if classes else int(pred.max()) + 1, palette)

    # an empty list means that no class is picked explicitly, so every class
    # but the background is drawn
    if class_ids is not None and len(class_ids) == 0:
        class_ids = None

    out = image.copy()
    if dim_background > 0:
        out = np.clip(
            out.astype(np.float32) * (1.0 - dim_background), 0, 255
        ).astype(np.uint8)

    for class_id in range(len(colors)):
        if class_ids is not None and class_id not in class_ids:
            continue

        mask = pred == class_id
        if not mask.any():
            continue

        color = colors[class_id]
        # the background is not drawn by default, so only the pixels of the
        # other classes are coloured
        if class_id == 0 and (class_ids is None or 0 not in class_ids):
            continue

        out = blend_mask(out, mask, color, ratio)

    return out


def add_legend(image, classes, palette=None, class_ids=None, ratio=0.7,
               position="top-left"):
    """Write a legend of the drawn classes on an image."""
    colors = get_palette(len(classes) if classes else 1, palette)

    # an empty list means that no class is picked explicitly, so every class
    # but the background is drawn
    if class_ids is not None and len(class_ids) == 0:
        class_ids = None

    entries = []
    for class_id, color in enumerate(colors):
        if class_ids is not None and class_id not in class_ids:
            continue
        if class_id == 0 and (class_ids is None or 0 not in class_ids):
            continue
        name = classes[class_id] if classes and class_id < len(classes) else f"class {class_id}"
        entries.append((name, color))

    if not entries:
        return image

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    pad = 6
    box_h = 22
    swatch = 16

    widths = [cv2.getTextSize(name, font, scale, thickness)[0][0] for name, _ in entries]
    box_w = max(widths) + swatch + pad * 3

    h, w = image.shape[:2]
    x0, y0 = pad, pad
    if position == "top-right":
        x0 = max(pad, w - box_w - pad)

    # a translucent panel behind the text; only the panel is blended, so the
    # rest of the picture stays untouched
    panel_h = box_h * len(entries) + pad
    region = image[y0:y0 + panel_h, x0:x0 + box_w]
    panel = np.zeros_like(region)
    cv2.rectangle(panel, (0, 0), (box_w - 1, panel_h - 1), (0, 0, 0), -1)
    alpha = 0.55
    image[y0:y0 + panel_h, x0:x0 + box_w] = cv2.addWeighted(
        panel, alpha, region, 1 - alpha, 0
    )

    y = y0 + pad
    for (name, color), text_w in zip(entries, widths):
        # the swatch shows the colour as it is blended into the image
        blended = tuple(int(c * ratio) for c in color)
        cv2.rectangle(
            image,
            (x0 + pad, y),
            (x0 + pad + swatch, y + swatch),
            blended,
            -1,
        )
        cv2.rectangle(
            image,
            (x0 + pad, y),
            (x0 + pad + swatch, y + swatch),
            color,
            1,
        )
        cv2.putText(
            image,
            name,
            (x0 + pad + swatch + pad, y + swatch - 3),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += box_h

    return image


def make_side_by_side(left, right, labels=None, gap=8, background=(255, 255, 255)):
    """Put two images next to each other.

    Args:
        left: the image on the left.
        right: the image on the right.
        labels: the titles of the two images.
        gap: the width of the band between the two images.

    Returns:
        The combined image.
    """
    if left.shape[0] != right.shape[0]:
        # resize the right image to the height of the left one
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(
            right,
            (max(1, int(right.shape[1] * scale)), left.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    h = max(left.shape[0], right.shape[0])
    gap_band = np.full((h, gap, 3), background, dtype=np.uint8)

    combined = np.concatenate([left, gap_band, right], axis=1)

    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        for index, text in enumerate(labels):
            if not text:
                continue
            x = index * (left.shape[1] + gap) + 8
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.rectangle(
                combined,
                (x - 4, 4),
                (x + tw + 4, th + 10),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                combined,
                text,
                (x, th + 6),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return combined


def make_figure(image, pred, label, classes, settings):
    """Build the picture of one image.

    The left half holds the label (or the original image when there is no
    label) and the right half the prediction.

    Args:
        image: the original image, a ``uint8`` BGR array.
        pred: the predicted class ids.
        label: the class ids of the label; None when there is no label.
        classes: the names of the classes.
        settings: the settings of the drawing (see ``resolve_settings``).

    Returns:
        The combined picture.
    """
    ratio = settings["ratio"]
    palette = settings["palette"]
    class_ids = settings["class_ids"]

    # the left half: the label when there is one, the image otherwise
    if label is not None:
        if label.shape[:2] != image.shape[:2]:
            label = cv2.resize(
                label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        left = overlay_prediction(
            image, label, classes=classes, palette=palette, ratio=ratio,
            class_ids=class_ids,
        )
        left = add_legend(
            left, classes, palette=palette, class_ids=class_ids, ratio=ratio
        )
        left_label = "label"
    else:
        left = image.copy()
        left_label = "image"

    # the right half: the prediction
    right = overlay_prediction(
        image, pred, classes=classes, palette=palette, ratio=ratio,
        class_ids=class_ids,
    )
    right = add_legend(
        right, classes, palette=palette, class_ids=class_ids, ratio=ratio
    )

    labels = settings.get("labels")
    if labels:
        titles = list(labels)
    else:
        titles = [left_label, "prediction"]

    return make_side_by_side(left, right, labels=titles, gap=settings["gap"])


def save_image(path, image, quality=95):
    """Write an image, creating the folders of the path when needed.

    Args:
        path: the path of the file.
        image: the picture, a ``uint8`` BGR array.
        quality: the quality of the jpg (1-100); it is ignored for the other
            formats.
    """
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    params = []
    if path.lower().endswith((".jpg", ".jpeg")):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, quality)))]

    if not cv2.imwrite(path, image, params):
        raise RuntimeError(f"failed to write the image: {path}")
    return path
