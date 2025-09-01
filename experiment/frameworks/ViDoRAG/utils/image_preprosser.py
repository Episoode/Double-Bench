from PIL import Image, ImageDraw
from collections import Counter

def concat_images_with_bbox(
    images,
    bboxs=None,
    arrangement=(3,3),
    scale=1.0,
    line_width=40,
    max_pixel=None
):
    """
    Concatenate a list of images according to a specified layout, draw bounding boxes, and return the new image along with bbox coordinates in the new image.

    Args:
    - images: list of str or PIL.Image. List of image file paths or PIL.Image objects to be concatenated.
    - bboxs: List[List[List[float]]], each element is a list of bboxes for one image,
             and each bbox is [y_min, x_min, y_max, x_max] with values in [0,1], relative to the image.
    - arrangement: tuple or str, arrangement of images. Can be (rows, columns) tuple,
                   or 'horizontal' or 'vertical'.
    - scale: float, scale factor for the final image.
    - line_width: int, width of the separator lines.
    - max_pixel: int or None, if set, will downscale so that total pixels do not exceed max_pixel.

    Returns:
    - final_img: PIL.Image, the concatenated image with bounding boxes.
    - new_bboxs: List[List[List[float]]], bbox coordinates in the final concatenated image,
                 also normalized between 0 and 1.
    """
    # Load images if paths are provided
    loaded_images = []
    for img in images:
        if isinstance(img, str):
            loaded_images.append(Image.open(img).convert('RGB'))
        else:
            loaded_images.append(img.convert('RGB'))
    images = loaded_images

    sizes = [image.size for image in images]
    size_counts = Counter(sizes)
    most_common_size = size_counts.most_common(1)[0][0]
    width, height = most_common_size

    # Resize all images to the most common size
    images_resized = [img.resize(most_common_size) for img in images]

    # Determine arrangement
    if isinstance(arrangement, tuple) and len(arrangement) == 2:
        rows, columns = arrangement
    elif arrangement == 'horizontal':
        rows = 1
        columns = len(images_resized)
    elif arrangement == 'vertical':
        rows = len(images_resized)
        columns = 1
    else:
        rows = 1
        columns = len(images_resized)

    total_cells = rows * columns

    # Pad or truncate images
    if len(images_resized) < total_cells:
        num_padding = total_cells - len(images_resized)
        blank_img = Image.new('RGB', most_common_size, color=(255,255,255))
        images_resized.extend([blank_img]*num_padding)
        if bboxs is not None:
            bboxs = bboxs + [[] for _ in range(num_padding)]
    elif len(images_resized) > total_cells:
        images_resized = images_resized[:total_cells]
        if bboxs is not None:
            bboxs = bboxs[:total_cells]
    elif bboxs is not None and len(bboxs) < total_cells:
        bboxs = bboxs + [[] for _ in range(total_cells - len(bboxs))]

    # Create new canvas
    total_width = columns * width + (columns - 1) * line_width
    total_height = rows * height + (rows - 1) * line_width
    new_img = Image.new('RGB', (total_width, total_height), color=(0,0,0))

    # Place images and record their placement
    placements = []
    for idx, img in enumerate(images_resized):
        row = idx // columns
        col = idx % columns
        x = col * (width + line_width)
        y = row * (height + line_width)
        new_img.paste(img, (x, y))
        placements.append((x, y, x + width, y + height))

    # Draw bounding boxes and calculate their new coordinates
    new_bboxs = []
    draw = ImageDraw.Draw(new_img)
    if bboxs is not None:
        for idx, bbox_list in enumerate(bboxs):
            x0, y0, x1, y1 = placements[idx]
            img_width = x1 - x0
            img_height = y1 - y0
            img_bboxs = []
            for bbox in bbox_list:
                # bbox: [y_min, x_min, y_max, x_max] in [0,1]
                y_min, x_min, y_max, x_max = bbox
                abs_x_min = x0 + int(x_min * img_width)
                abs_y_min = y0 + int(y_min * img_height)
                abs_x_max = x0 + int(x_max * img_width)
                abs_y_max = y0 + int(y_max * img_height)
                # Draw rectangle
                draw.rectangle([abs_x_min, abs_y_min, abs_x_max, abs_y_max], outline=(255,0,0), width=3)
                # Save relative bbox in final image
                norm_y_min = abs_y_min / total_height
                norm_x_min = abs_x_min / total_width
                norm_y_max = abs_y_max / total_height
                norm_x_max = abs_x_max / total_width
                img_bboxs.append([norm_y_min, norm_x_min, norm_y_max, norm_x_max])
            new_bboxs.append(img_bboxs)
    else:
        new_bboxs = [[] for _ in range(total_cells)]

    # Scale the final image if needed
    if max_pixel is not None:
        total_pixels = total_width * total_height
        scale = (max_pixel / total_pixels) ** 0.5
        scaled_width = int(total_width * scale)
        scaled_height = int(total_height * scale)
        final_img = new_img.resize((scaled_width, scaled_height))
        # Adjust bbox coordinates for scaling
        scaled_bboxs = []
        for img_bboxs in new_bboxs:
            scaled_img_bboxs = []
            for bbox in img_bboxs:
                norm_y_min, norm_x_min, norm_y_max, norm_x_max = bbox
                scaled_img_bboxs.append([norm_y_min, norm_x_min, norm_y_max, norm_x_max])
            scaled_bboxs.append(scaled_img_bboxs)
        return final_img, scaled_bboxs
    else:
        if scale != 1.0:
            scaled_width = int(total_width * scale)
            scaled_height = int(total_height * scale)
            final_img = new_img.resize((scaled_width, scaled_height))
            # Adjust bbox coordinates for scaling
            scaled_bboxs = []
            for img_bboxs in new_bboxs:
                scaled_img_bboxs = []
                for bbox in img_bboxs:
                    norm_y_min, norm_x_min, norm_y_max, norm_x_max = bbox
                    scaled_img_bboxs.append([norm_y_min, norm_x_min, norm_y_max, norm_x_max])
                scaled_bboxs.append(scaled_img_bboxs)
            return final_img, scaled_bboxs
        else:
            return new_img, new_bboxs