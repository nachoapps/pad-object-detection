r"""Extract orbs from annotated images, prep for gcs upload.

Example usage:
    python scripts/create_orb_images.py \
        --data_dir=images \
        --annotations_dir=annotations \
        --output_path=data \
        --bucket_path=my_bucket/my_dir
"""
import PIL.Image
import hashlib
import logging
from lxml import etree
import os
import csv

import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', 'images',
                    '(Relative) path to images directory')
flags.DEFINE_string('annotations_dir', 'annotations',
                    '(Relative) path to annotations directory')
flags.DEFINE_string('output_path', '', 'Path to dump orbs into')
flags.DEFINE_string('bucket_path', 'rpad-discord-vcm', 'GCS Path')
FLAGS = flags.FLAGS

RANDOM_SEED = 4242
VALIDATION_PCT = .1
MAX_IMG_DIM = 1024.0

ORB_SIZE=64
ORB_EXPANSION_PCT=.1


def extract_orb_annotations(data):
    return [x for x in data['object'] if 'orb' in x['name']]


def load_image(data, image_dir):
    input_path = os.path.join(image_dir, data['filename'])

    if not 'object' in data:
        raise ValueError('no objects found in {}'.format(input_path))

    image = PIL.Image.open(input_path)
    if image.format not in ('JPEG', 'PNG'):
        raise ValueError('Image format not JPEG/PNG')

    return image


def process_orbs(data, image_dir, output_dir):
    """Extracts orb images from annotated files and saves them to disk."""
    image = load_image(data, image_dir)

    image_width, image_height = image.size
    orb_annotations = extract_orb_annotations(data)
    for obj in orb_annotations:
        class_text = obj['name']

        bb = obj['bndbox']
        xmin = int(bb['xmin'])
        xmax = int(bb['xmax'])
        ymin = int(bb['ymin'])
        ymax = int(bb['ymax'])

        def save_orb(xmin, ymin, xmax, ymax):
            orb_img = image.crop((xmin, ymin, xmax, ymax))
            orb_img = orb_img.resize((ORB_SIZE, ORB_SIZE), PIL.Image.ANTIALIAS)

            output_color_path = os.path.join(output_dir, class_text)
            os.makedirs(output_color_path, exist_ok=True)

            key = hashlib.md5(orb_img.tobytes()).hexdigest()
            output_orb_path = os.path.join(output_color_path, key + '.png')

            orb_img.save(output_orb_path)

        # Save the directly annotated orb values.
        save_orb(xmin, ymin, xmax, ymax)

        # Save a slightly expanded orb to allow for some error
        x_offset = ORB_EXPANSION_PCT * (xmax - xmin)
        y_offset = ORB_EXPANSION_PCT * (ymax - ymin)
        xmin = max(0, xmin - x_offset)
        xmax = min(image_width, xmax + x_offset)
        ymin = max(0, ymin - y_offset)
        ymax = min(image_height, ymax + y_offset)
        save_orb(xmin, ymin, xmax, ymax)


def process_images(data, image_dir, output_dir):
    """Reformats input images and creates the annotation csv."""
    image = load_image(data, image_dir)
    orb_annotations = extract_orb_annotations(data)

    orb_count = len(orb_annotations)
    if orb_count not in [4*5, 5*6, 6*7]:
        print('unexpected orb count of', orb_count, 'for', data['filename'], 'skipping it')
        return []

    max_dim = float(max(image.size[0], image.size[1]))
    if max_dim > MAX_IMG_DIM:
        scale_factor = 1.0 - (max_dim - MAX_IMG_DIM) / max_dim
        if image.size[0] > image.size[1]:
            new_width = MAX_IMG_DIM
            new_height = image.size[1] * scale_factor
        else:
            new_width = image.size[0] * scale_factor
            new_height = MAX_IMG_DIM

        image = image.resize((int(new_width), int(new_height)), PIL.Image.ANTIALIAS)

    img_output_path = os.path.join(output_dir, 'corrected_images')
    os.makedirs(img_output_path, exist_ok=True)

    key = hashlib.md5(image.tobytes()).hexdigest()
    output_img_path = os.path.join(img_output_path, key + '.png')
    image.save(output_img_path)

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    results = []

    for obj in orb_annotations:
        class_text = 'orb'

        xmin = float(obj['bndbox']['xmin']) / width
        ymin = float(obj['bndbox']['ymin']) / height
        xmax = float(obj['bndbox']['xmax']) / width
        ymax = float(obj['bndbox']['ymax']) / height
        results.append(('UNASSIGNED', output_img_path, class_text, xmin, ymin, xmax, ymax))

    return results



# Hacked in from object_detection library
def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.
    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
      Python dictionary holding XML contents.
    """
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def load_annotations(annotation_file):
    with tf.gfile.GFile(annotation_file, 'r') as fid:
        xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        return recursive_parse_xml_to_dict(xml)['annotation']


def do_orb_processing(image_dir, orb_output_path, annotation_files, orb_bucket_path):
    for idx, annotation_file in enumerate(annotation_files):
        print(annotation_file)
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(annotation_files))

        data = load_annotations(annotation_file)
        process_orbs(data, image_dir, orb_output_path)

    orb_output_csv = os.path.join(orb_output_path, 'orb_data.csv')
    last_output_path = os.path.basename(orb_output_path)
    with open(orb_output_csv, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for label in os.listdir(orb_output_path):
            label_dir = os.path.join(orb_output_path, label)
            if not os.path.isdir(label_dir):
                continue
            for orb_filename in os.listdir(label_dir):
                final_label_dir = os.path.join(last_output_path, label)
                gcs_file_path = 'gs://{}/{}/{}'.format(orb_bucket_path, final_label_dir, orb_filename)
                csv_writer.writerow([gcs_file_path, label])

    print_gsutil_help(orb_output_path, orb_bucket_path, last_output_path, 'orb_data.csv')


def pr(num):
    return round(num, 4)

def do_screen_processing(image_dir, screen_output_path, annotation_files, screen_bucket_path):
    os.makedirs(screen_output_path, exist_ok=True)
    screen_output_csv = os.path.join(screen_output_path, 'screen_data.csv')
    last_output_path = os.path.basename(screen_output_path)
    with open(screen_output_csv, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for idx, annotation_file in enumerate(annotation_files):
            print(annotation_file)
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(annotation_files))

            data = load_annotations(annotation_file)
            results = process_images(data, image_dir, screen_output_path)

            for item in results:
                item_path = item[1].replace(screen_output_path + '/', '')
                gcs_file_path = 'gs://{}/{}/{}'.format(screen_bucket_path, last_output_path, item_path)
                csv_writer.writerow([item[0], gcs_file_path, item[2],
                                     pr(item[3]), pr(item[4]), '', '',
                                     pr(item[5]), pr(item[6]), '', ''])

    print_gsutil_help(screen_output_path, screen_bucket_path, last_output_path, 'screen_data.csv')


def print_gsutil_help(local_path, bucket_path, bucket_folder, csv_file_name):
    print('you probably want to execute:')
    print('  gsutil -m rsync -r -c {} gs://{}/{}'.format(local_path, bucket_path, bucket_folder))

    print('consider running this command to force-sync the remote dir:')
    print('  gsutil -m rsync -r -c -d {} gs://{}/{}'.format(local_path, bucket_path, bucket_folder))

    print('the CSV file will be available at:')
    print('  gs://{}/{}/{}'.format(bucket_path, bucket_folder, csv_file_name))


def main(_):
    image_dir = FLAGS.data_dir
    annotation_files = tf.gfile.Glob(os.path.join(FLAGS.annotations_dir, '*.xml'))
    logging.info('Processing %s images.', len(annotation_files))

    orb_output_path = os.path.join(FLAGS.output_path, 'extracted_orb_images')
    orb_bucket_path = os.path.join(FLAGS.bucket_path, 'orbs')
    print('starting orb processing')
    do_orb_processing(image_dir, orb_output_path, annotation_files, orb_bucket_path)

    print('starting screen processing')
    screen_output_path = os.path.join(FLAGS.output_path, 'orbs_in_screens')
    screen_bucket_path = os.path.join(FLAGS.bucket_path, 'orbs')
    do_screen_processing(image_dir, screen_output_path, annotation_files, screen_bucket_path)


if __name__ == '__main__':
    tf.app.run()
