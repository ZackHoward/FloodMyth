"""Create a vertical frame sequence for the FloodMyth motif barcode reveal.

Run from the repository root:

    python scripts/create_barcode_animation_frames.py

The script writes numbered PNG frames to:

    outputs/animation_frames/barcode_reveal_v0_1/

Video editor import note:
In CapCut, DaVinci Resolve, Premiere, or similar tools, import the generated
PNG files as an image sequence. Set the sequence frame rate to FPS below.
"""

from pathlib import Path
import math
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


FPS = 24
TOTAL_SECONDS = 5
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920
OUTPUT_DIR = Path("outputs") / "animation_frames" / "barcode_reveal_v0_1"

DATA_PATH = Path("outputs") / "motif_matrix_numeric_v0_1.csv"

BACKGROUND = "#050816"
PRESENT_COLOR = "#00E5FF"
PARTIAL_COLOR = "#F2A900"
ABSENT_COLOR = "#101820"
UNCLEAR_COLOR = "#5A5F66"
GRID_COLOR = "#FFFFFF"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT_COLOR = "#CBD5E1"
HIGHLIGHT_COLOR = "#39FF88"
OUTLIER_COLOR = "#F2A900"
SCAN_COLOR = "#9AF7FF"


MOTIF_GROUPS = {
    "Cause": [
        "moral_cause",
        "noise_or_overpopulation_cause",
        "failed_creation_cause",
        "primordial_or_cosmic_disorder",
        "world_age_cycle",
        "divine_decision_to_flood_or_destroy",
        "council_or_multiple_deities",
        "flood_as_natural_or_sea_overflow",
        "blood_flood",
        "flood_control_governance_problem",
    ],
    "Warning + Survivor": [
        "chosen_survivor_or_group",
        "survivor_righteous_pious_religious",
        "survivor_is_king_priest_ruler",
        "survivor_couple",
        "family_or_household_saved",
        "animal_or_supernatural_helper_warning",
        "divine_or_supernatural_warning",
        "warning_secret_indirect_wall",
        "dream_warning",
        "specific_warning_timing",
    ],
    "Vessel + Refuge": [
        "boat_or_vessel",
        "non_boat_container_or_tree_chest_vessel",
        "construction_instructions",
        "specific_vessel_dimensions",
        "pitch_or_waterproofing",
        "food_stored_or_rationed",
        "animals_preserved_or_co_survive",
        "civilization_knowledge_or_craftsmen_preserved",
        "mountain_landing_or_refuge",
        "waters_recede_or_subside",
    ],
    "Catastrophe": [
        "storm_or_rain",
        "water_from_above_and_below_or_cosmic_sources",
        "sea_lake_overflow",
        "totalizing_flood_language",
        "regional_flood_language",
        "humanity_destroyed_or_reset",
        "nonhuman_or_giant_population_destroyed",
        "animals_or_objects_rebel",
        "transformation_after_catastrophe",
        "bird_test",
    ],
    "Aftermath": [
        "post_flood_sacrifice_or_offering",
        "divine_smells_or_accepts_offering",
        "post_flood_offering_misdirected_or_inverted",
        "divine_regret_grief_conflict_after_flood",
        "survivor_reward_or_blessing",
        "covenant_promise_or_rainbow",
        "immortality_or_godlike_reward",
        "new_rules_or_order_after_flood",
        "population_control_after_flood",
        "humanity_restored_after_flood",
        "ritual_recreation_after_flood",
        "stones_bones_blood_or_material_recreation",
        "civilization_restored_after_flood",
        "hydraulic_engineering_or_public_works",
        "animal_body_or_species_origin_motif",
        "source_biblical_influence_caution",
    ],
}


STORY_LABEL_MAP = {
    "Genesis Flood / Noah": "Genesis\nNoah",
    "Gilgamesh Tablet XI / Utanapishtim": "Gilgamesh\nUtanapishtim",
    "Atrahasis Flood": "Atrahasis",
    "Eridu Genesis / Ziusudra": "Eridu Genesis\nZiusudra",
    "Deucalion and Pyrrha": "Deucalion\nPyrrha",
    "Manu and the Fish": "Manu\nFish",
    "Yu Controls the Flood": "Yu\nControls Flood",
    "Popol Vuh / Destruction of the Wooden People": "Popol Vuh\nWooden People",
    "Huarochiri / The Llama Warns of the Flood": "Huarochiri\nLlama",
    "Huarochirí / The Llama Warns of the Flood": "Huarochiri\nLlama",
    "Nahui Atl / Tata and Nene": "Nahui Atl\nTata + Nene",
    "Bergelmir and the Blood Flood": "Bergelmir\nBlood Flood",
    "Nuʻu and the Hawaiian Flood": "Nuʻu\nHawaiian Flood",
}

CLASSIC_CLUSTER = [
    "Genesis Flood / Noah",
    "Gilgamesh Tablet XI / Utanapishtim",
    "Atrahasis Flood",
    "Eridu Genesis / Ziusudra",
]

OUTLIER_ROWS = [
    "Yu Controls the Flood",
    "Popol Vuh / Destruction of the Wooden People",
    "Nahui Atl / Tata and Nene",
    "Bergelmir and the Blood Flood",
]


def ease_in_out(value):
    """Smooth animation progress from 0 to 1."""
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def hex_to_rgb(hex_color, alpha=255):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def mix_color(hex_color, alpha):
    red, green, blue, _ = hex_to_rgb(hex_color)
    return (red, green, blue, max(0, min(255, int(alpha * 255))))


def blend_color(foreground, background, amount):
    """Blend two hex colors and return an opaque RGB tuple."""
    amount = max(0.0, min(1.0, amount))
    fg = hex_to_rgb(foreground)[:3]
    bg = hex_to_rgb(background)[:3]
    return tuple(int(bg[i] + (fg[i] - bg[i]) * amount) for i in range(3))


def load_font(size, bold=False):
    """Use common bundled fonts when available, then fall back to Pillow."""
    font_names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]

    for font_name in font_names:
        font_path = Path(font_name)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)

    return ImageFont.load_default(size=size)


def text_size(draw, text, font):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_multiline(draw, xy, text, font, fill, spacing=4):
    x, y = xy
    width, height = text_size(draw, text, font)
    draw.multiline_text(
        (x - width / 2, y - height / 2),
        text,
        font=font,
        fill=fill,
        anchor=None,
        align="center",
        spacing=spacing,
    )


def draw_right_aligned_multiline(draw, xy, text, font, fill, spacing=4):
    x, y = xy
    width, height = text_size(draw, text, font)
    draw.multiline_text(
        (x - width, y - height / 2),
        text,
        font=font,
        fill=fill,
        anchor=None,
        align="right",
        spacing=spacing,
    )


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    metadata_cols = [
        "story_id",
        "story_name",
        "culture",
        "region",
        "primary_source",
        "source_type",
        "story_type",
        "overall_confidence",
        "source_caution",
        "notes",
        "include_in_core_analysis",
        "analysis_group",
        "recommended_first_visual",
        "score_notes",
    ]
    score_cols = [
        "present_motif_score",
        "uncertain_or_partial_count",
        "core_survivor_package_score",
        "expanded_flood_catastrophe_score",
    ]

    metadata_cols = [col for col in metadata_cols if col in df.columns]
    score_cols = [col for col in score_cols if col in df.columns]
    motif_cols = [
        col for col in df.columns if col not in metadata_cols and col not in score_cols
    ]

    ordered_motif_cols = [
        motif for group in MOTIF_GROUPS.values() for motif in group if motif in motif_cols
    ]

    df["short_label"] = df["story_name"].map(STORY_LABEL_MAP).fillna(df["story_name"])
    matrix = df[ordered_motif_cols].astype(float).to_numpy()

    return df, matrix, ordered_motif_cols


def color_for_value(value, opacity=1.0):
    if np.isnan(value):
        return mix_color(UNCLEAR_COLOR, opacity)
    if value >= 0.75:
        return mix_color(PRESENT_COLOR, opacity)
    if value >= 0.25:
        return mix_color(PARTIAL_COLOR, opacity)
    return mix_color(ABSENT_COLOR, opacity)


def draw_glow_rectangle(image, box, color, alpha, width=4, blur=10):
    """Draw a soft glow and a crisp outline around a row or cluster."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow = ImageDraw.Draw(overlay)
    rgba = mix_color(color, alpha)
    for inset in range(0, width * 2, 2):
        glow.rectangle(
            (
                box[0] - inset,
                box[1] - inset,
                box[2] + inset,
                box[3] + inset,
            ),
            outline=rgba,
            width=width,
        )

    blurred = overlay.filter(ImageFilter.GaussianBlur(blur))
    image.alpha_composite(blurred)

    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=mix_color(color, min(1.0, alpha + 0.25)), width=width)


def draw_frame(
    frame_index,
    df,
    matrix,
    ordered_motif_cols,
    fonts,
    group_geometry,
    row_geometry,
):
    total_frames = FPS * TOTAL_SECONDS
    progress = frame_index / max(1, total_frames - 1)

    image = Image.new("RGBA", (FRAME_WIDTH, FRAME_HEIGHT), hex_to_rgb(BACKGROUND))
    draw = ImageDraw.Draw(image)

    # The layout mirrors the static YouTube barcode, but reserves open space
    # for CapCut titles/subtitles outside this generated asset.
    left = 310
    right = 40
    top = 390
    bottom = 300
    barcode_width = FRAME_WIDTH - left - right
    barcode_height = FRAME_HEIGHT - top - bottom
    row_count, col_count = matrix.shape
    cell_width = barcode_width / col_count
    cell_height = barcode_height / row_count

    label_alpha = ease_in_out(progress / 0.12)
    reveal_alpha = ease_in_out((progress - 0.10) / 0.46)
    reveal_x = left + barcode_width * reveal_alpha

    # Subtle background guide lines establish the scanner feel before cells appear.
    for row in range(row_count + 1):
        y = top + row * cell_height
        draw.line(
            [(left, y), (left + barcode_width, y)],
            fill=blend_color(GRID_COLOR, BACKGROUND, 0.08 + 0.06 * label_alpha),
            width=1,
        )

    for boundary_x in group_geometry["boundaries"]:
        x = left + boundary_x * cell_width
        draw.line(
            [(x, top), (x, top + barcode_height)],
            fill=blend_color(GRID_COLOR, BACKGROUND, 0.13 + 0.12 * label_alpha),
            width=1,
        )

    # Story labels fade in first, before the barcode scanner passes over columns.
    for row_index, label in enumerate(df["short_label"]):
        y = top + row_index * cell_height + cell_height / 2
        draw_right_aligned_multiline(
            draw,
            (left - 20, y),
            label,
            fonts["row"],
            blend_color(TEXT_COLOR, BACKGROUND, 0.28 + 0.72 * label_alpha),
            spacing=6,
        )

    # Top group labels stay clean and readable, with the longer names wrapped.
    group_label_map = {
        "Warning + Survivor": "Warning\n+ Survivor",
        "Vessel + Refuge": "Vessel\n+ Refuge",
    }
    # These labels name the motif bands, but the text is wider than the bands.
    # Use hand-tuned horizontal anchors so the words breathe on a phone screen.
    group_label_positions = {
        "Cause": 0.08,
        "Warning + Survivor": 0.30,
        "Vessel + Refuge": 0.50,
        "Catastrophe": 0.72,
        "Aftermath": 0.92,
    }
    for group_name in group_geometry["centers"]:
        label = group_label_map.get(group_name, group_name)
        x = left + barcode_width * group_label_positions[group_name]
        draw_centered_multiline(
            draw,
            (x, top - 82),
            label,
            fonts["group"],
            blend_color(MUTED_TEXT_COLOR, BACKGROUND, 0.30 + 0.70 * label_alpha),
            spacing=0,
        )

    # Draw the barcode cells only up to the current scanner position.
    for row in range(row_count):
        for col in range(col_count):
            x0 = left + col * cell_width
            x1 = left + (col + 1) * cell_width
            if reveal_x <= x0:
                continue

            visible_fraction = min(1.0, max(0.0, (reveal_x - x0) / cell_width))
            cell_alpha = 0.22 + 0.78 * visible_fraction
            y0 = top + row * cell_height
            y1 = top + (row + 1) * cell_height

            draw.rectangle(
                (round(x0), round(y0), round(x1), round(y1)),
                fill=color_for_value(matrix[row, col], opacity=cell_alpha),
            )

    # Redraw separators on top of cells for a crisp data-grid look.
    for row in range(row_count + 1):
        y = top + row * cell_height
        draw.line(
            [(left, y), (left + barcode_width, y)],
            fill=(255, 255, 255, 34),
            width=1,
        )

    for boundary_x in group_geometry["boundaries"]:
        x = left + boundary_x * cell_width
        draw.line([(x, top), (x, top + barcode_height)], fill=(255, 255, 255, 85), width=2)

    # Scanner line and glow during the reveal.
    if 0.01 < reveal_alpha < 0.995:
        line_x = int(reveal_x)
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        glow_draw.rectangle(
            (line_x - 7, top - 8, line_x + 7, top + barcode_height + 8),
            fill=mix_color(SCAN_COLOR, 0.28),
        )
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)))
        draw.line(
            [(line_x, top - 8), (line_x, top + barcode_height + 8)],
            fill=mix_color(SCAN_COLOR, 0.95),
            width=3,
        )

    # Legend appears as the reveal finishes, centered under the barcode.
    legend_alpha = ease_in_out((progress - 0.38) / 0.18)
    if legend_alpha > 0.01:
        legend_rows = [
            [(160, "Present", PRESENT_COLOR), (585, "Partial", PARTIAL_COLOR)],
            [(160, "Absent", ABSENT_COLOR), (585, "Unclear / ambiguous", UNCLEAR_COLOR)],
        ]
        for row_offset, legend_row in enumerate(legend_rows):
            legend_y = FRAME_HEIGHT - 190 + row_offset * 78
            for legend_x, label, color in legend_row:
                draw.rectangle(
                    (legend_x, legend_y - 22, legend_x + 66, legend_y + 22),
                    fill=mix_color(color, legend_alpha),
                )
                draw.text(
                    (legend_x + 86, legend_y),
                    label,
                    font=fonts["legend"],
                    fill=mix_color(TEXT_COLOR, legend_alpha),
                    anchor="lm",
                )

    # Highlight the classic survivor-flood cluster after the barcode is revealed.
    cluster_start = 0.58
    if progress >= cluster_start:
        pulse = 0.65 + 0.35 * math.sin(frame_index * 0.22)
        alpha = ease_in_out((progress - cluster_start) / 0.14) * pulse
        first_row = row_geometry["classic_first"]
        last_row = row_geometry["classic_last"]
        box = (
            left - 8,
            top + first_row * cell_height - 8,
            left + barcode_width + 8,
            top + (last_row + 1) * cell_height + 8,
        )
        draw_glow_rectangle(image, box, HIGHLIGHT_COLOR, alpha, width=4, blur=12)

    # Then pulse the selected outliers one at a time.
    outlier_start = 0.70
    outlier_end = 0.96
    if outlier_start <= progress <= outlier_end:
        outlier_progress = (progress - outlier_start) / (outlier_end - outlier_start)
        active_index = min(len(OUTLIER_ROWS) - 1, int(outlier_progress * len(OUTLIER_ROWS)))
        local = outlier_progress * len(OUTLIER_ROWS) - active_index
        pulse = math.sin(math.pi * local)
        row = row_geometry["outliers"][active_index]
        box = (
            left - 8,
            top + row * cell_height - 7,
            left + barcode_width + 8,
            top + (row + 1) * cell_height + 7,
        )
        draw_glow_rectangle(image, box, OUTLIER_COLOR, max(0.0, pulse), width=4, blur=14)

    return image.convert("RGB")


def build_geometry(df, ordered_motif_cols):
    group_centers = {}
    group_boundaries = [0]
    start = 0
    for group_name, group_cols in MOTIF_GROUPS.items():
        valid_cols = [col for col in group_cols if col in ordered_motif_cols]
        if not valid_cols:
            continue
        end = start + len(valid_cols)
        group_centers[group_name] = start + (len(valid_cols) - 1) / 2
        group_boundaries.append(end)
        start = end

    story_to_row = {story_name: idx for idx, story_name in enumerate(df["story_name"])}
    classic_rows = [story_to_row[name] for name in CLASSIC_CLUSTER]
    outlier_rows = [story_to_row[name] for name in OUTLIER_ROWS]

    return (
        {"centers": group_centers, "boundaries": group_boundaries},
        {
            "classic_first": min(classic_rows),
            "classic_last": max(classic_rows),
            "outliers": outlier_rows,
        },
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input CSV: {DATA_PATH}")

    df, matrix, ordered_motif_cols = prepare_data()
    group_geometry, row_geometry = build_geometry(df, ordered_motif_cols)

    # Recreate the folder so old frames from longer previous runs do not linger.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fonts = {
        "row": load_font(31, bold=True),
        "group": load_font(24, bold=True),
        "legend": load_font(35, bold=False),
    }

    total_frames = FPS * TOTAL_SECONDS
    print(f"Generating {total_frames} frames at {FRAME_WIDTH}x{FRAME_HEIGHT}...")
    print(f"Output folder: {OUTPUT_DIR}")

    for frame_index in range(total_frames):
        frame = draw_frame(
            frame_index,
            df,
            matrix,
            ordered_motif_cols,
            fonts,
            group_geometry,
            row_geometry,
        )
        frame_path = OUTPUT_DIR / f"barcode_reveal_{frame_index:04d}.png"
        frame.save(frame_path, optimize=True)

        if frame_index % FPS == 0:
            print(f"  wrote frame {frame_index:04d}")

    print("Done.")


if __name__ == "__main__":
    main()
