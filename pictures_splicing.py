import os
import sys
import time
from functools import wraps
from threading import Timer

from PIL import Image
from PIL import ImageGrab

SOURCE_FILES_PATH = 'pictures_splicing/'
SNAPSHOT_FILES_PATH = 'pictures_splicing/snapshot/'
OUTPUT_FILE_PATH = 'pictures_splicing/output/'
LINE_IDENTICAL_JUMP = 2
LINE_IDENTICAL_RANGE = (0, 0.75)
SUPERPOSITION_THRESHOLD_PROPORTION = 1.0/7


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


class RepeatedTimer(object):

    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.function = function
        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        self.is_running = False

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def line_identical(pix_1, y_1, pix_2, y_2, width, jump_y):
    start, end = LINE_IDENTICAL_RANGE
    i = int(width * start)
    border = width * end
    while i < border:
        if pix_1[i, y_1] != pix_2[i, y_2]:
            return False

        i += jump_y

    return True


def get_images_from_files():
    def legal(filename):
        filepath = os.path.join(SOURCE_FILES_PATH, filename)
        if os.path.isfile(filepath):
            return True

        return False

    files = [os.path.join(SOURCE_FILES_PATH, filename) for filename in os.listdir(SOURCE_FILES_PATH) if legal(filename)]
    files.sort(key=lambda file: os.path.getmtime(file))

    images = []
    for file in files:
        try:
            image = Image.open(file)
        except IOError:
            continue

        print(file)
        images.append(image)

    return images


def get_clipboard_images(images):
    image = ImageGrab.grabclipboard()
    if image:
        last_index = len(images) - 1
        if last_index < 0:
            print('Got a snapshot from clipboard')
            images.append(image)
        else:
            if image != images[last_index]:
                print('Got a snapshot from clipboard')
                images.append(image)


def make_output_file_path_legal():
    if not os.path.exists(OUTPUT_FILE_PATH):
        os.makedirs(OUTPUT_FILE_PATH)


def make_snapshot_files_legal():
    if not os.path.exists(SNAPSHOT_FILES_PATH):
        os.makedirs(SNAPSHOT_FILES_PATH)


def save_recording_images(images):
    make_snapshot_files_legal()

    for i, image in enumerate(images):
        image.save(os.path.join(SNAPSHOT_FILES_PATH, '{}.png'.format(i)))


def save_long_shot(long_shot):
    if long_shot:
        make_output_file_path_legal()
        long_shot.save(os.path.join(OUTPUT_FILE_PATH, 'long_shot.png'))
        print('Saved long_shot.png in ' + OUTPUT_FILE_PATH)


def get_recording_images():
    images = []
    repeated_timer = RepeatedTimer(1, get_clipboard_images, images)
    repeated_timer.start()

    if input('Press "Enter" to stop\n') == '':
        repeated_timer.stop()

    save_recording_images(images)
    return images


def get_merge_long_shot_size(parameters):
    width = 0
    height = 0
    for parameter in parameters:
        height += parameter['end'] - parameter['start']
        width = parameter['width'] if parameter['width'] > width else width

    return width, height


@timethis
def search_superposition(image_1, image_2, index, parameters):
    w_1, h_1 = image_1.size
    w_2, h_2 = image_2.size

    pix_1 = image_1.load()
    pix_2 = image_2.load()

    below_offset = int(max(h_1, h_2) * SUPERPOSITION_THRESHOLD_PROPORTION)

    line_search_width = min(w_1, w_2)
    superposition_threshold = int(SUPERPOSITION_THRESHOLD_PROPORTION * min(h_1, h_2))
    jump_lines = int(superposition_threshold * 3.0 / 4)

    for y_1 in range(parameters[index]['start'], h_1-jump_lines-1, jump_lines):
        for y_2 in range(0, h_2-1, 1):
            if line_identical(pix_1, y_1, pix_2, y_2, line_search_width, LINE_IDENTICAL_JUMP):
                # print('Found')
                identical_lines = 0
                
                image_1_end = y_1
                image_2_start = y_2
                
                # Search prev
                for i in range(1, jump_lines, 1):
                    y_1_prev = y_1 - i
                    y_2_prev = y_2 - i
                    if min(y_1_prev, y_2_prev) <= 0:
                        break

                    if line_identical(pix_1, y_1_prev, pix_2, y_2_prev, line_search_width, 1):
                        # print('p')
                        identical_lines += 1
                    else:
                        break

                # Search next
                for i in range(0, superposition_threshold, 1):
                    y_1_next = y_1 + i
                    y_2_next = y_2 + i
                    if line_identical(pix_1, y_1_next, pix_2, y_2_next, line_search_width, 1):
                        # print('n', y_1_next)
                        identical_lines += 1
                        image_1_end = y_1_next
                        image_2_start = y_2_next
                    else:
                        break

                if identical_lines >= superposition_threshold:
                    parameters[index]['end'] = image_1_end
                    parameters[index]['width'] = w_1
                    parameters[index+1]['start'] = image_2_start
                    parameters[index+1]['end'] = h_2
                    parameters[index+1]['width'] = w_2
                    return True
                else:
                    # print('R: ', identical_lines, superposition_threshold)
                    pass

    # print('End ', y_1, y_2)
    return False


@timethis
def paste_merged_images(long_shot, images, parameters):
    reference_y = 0
    for i, image in enumerate(images):
        start, end, width = parameters[i]['start'], parameters[i]['end'], parameters[i]['width']
        height = end - start
        region = image.crop((0, start, width, end))
        box = (0, reference_y, width, reference_y+height)
        long_shot.paste(region, box)
        
        reference_y += height

    return long_shot


@timethis
def merge_images(images):
    make_output_file_path_legal()

    if len(images) == 1:
        return images[0]

    pretreated_images = [image.convert('L') for image in images]

    progress_rate = 0
    total = len(pretreated_images) - 1
    parameter = {
        'start': 0,
        'end': 0,
        'width': 0,
    }
    parameters = [dict(parameter) for _ in pretreated_images]

    for image_1, image_2 in zip(pretreated_images, pretreated_images[1:]):
        print('{} / {} ...'.format(progress_rate+1, total))
        success = search_superposition(image_1, image_2, progress_rate, parameters)
        if not success:
            print('Err: superposition not found! ({} & {})'.format(progress_rate+1, progress_rate+2))
            return None
        progress_rate += 1

    for parameter in parameters:
        print(parameter)

    width, height = get_merge_long_shot_size(parameters)
    print('Size: ({}, {})'.format(width, height))
    long_shot = Image.new('RGB', (width, height), 'white')

    long_shot = paste_merged_images(long_shot, images, parameters)
    save_long_shot(long_shot)


def get_splicing_long_shot_size(images):
    width = 0
    height = 0
    for image in images:
        w, h = image.size
        height += h
        width = w if w > width else width

    return width, height


@timethis
def splicing_images(images):
    make_output_file_path_legal()

    width, height = get_splicing_long_shot_size(images)
    long_shot = Image.new('RGB', (width, height), 'white')

    reference_y = 0
    for image in images:
        w, h = image.size
        box = (0, reference_y, w, reference_y+h)
        long_shot.paste(image, box)
        
        reference_y += h

    return long_shot


@timethis
def merge_files_handler():
    images = get_images_from_files()

    if not images:
        print('No images found')
    else:
        long_shot = merge_images(images)
        save_long_shot(long_shot)


@timethis
def merge_recording_handler():
    images = get_recording_images()
    
    if not images:
        print('No images found')
    else:
        long_shot = merge_images(images)
        save_long_shot(long_shot)


@timethis
def splicing_files_handler():
    images = get_images_from_files()

    if not images:
        print('No images found')
    else:
        long_shot = splicing_images(images)
        save_long_shot(long_shot)


@timethis
def splicing_recording_handler():
    images = get_recording_images()
    
    if not images:
        print('No images found')
    else:
        long_shot = splicing_images(images)
        save_long_shot(long_shot)


def help():
    print('Usage:\n'
        '    -mf: merge from files\n'
        '    -mr: merge from clipboard recording\n'
        '    -sf: simple splicing from files\n'
        '    -sr: simple splicing from clipboard recording\n'
        '    -help: show usage\n')


@timethis
def main():
    if len(sys.argv) == 2:
        parameter = sys.argv[1]

        if parameter in ['-mf', 'mf']:
            print('Merge from files')
            merge_files_handler()
        elif parameter in ['-mr', 'mr']:
            print('Merge from clipboard recording')
            merge_recording_handler()
        elif parameter in ['-sf', 'sf']:
            print('Simple splicing from files')
            splicing_files_handler()
        elif parameter in ['-sr', 'sr']:
            print('Simple splicing from clipboard recording')
            splicing_recording_handler()
        else:
            help()
    else:
        help()  


if __name__ == '__main__':
    main()
