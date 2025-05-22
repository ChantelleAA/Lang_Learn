from PIL import Image

def save_image(image, path="output/uploaded_image.jpg"):
    image.save(path)
    return path

def print_summary(results):
    for item in results:
        print(f"Object: {item['object']}")
        print(f"Sentence: {item['sentence']}")
        print(f"Valid Grammar: {item['valid']}")
        print(f"Complexity Score: {item['complexity']:.2f}")
        print("-" * 50)
