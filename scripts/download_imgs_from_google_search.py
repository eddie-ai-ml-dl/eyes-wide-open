from icrawler.builtin import GoogleImageCrawler
import os

def download_images(root_dir, query, num_images):
    folder_name = query.replace(" ", "_")  # Replace spaces with underscores for folder name
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    google_crawler = GoogleImageCrawler(storage={'root_dir': folder_path})
    google_crawler.crawl(keyword=query, max_num=num_images)


if __name__ == "__main__":
    root_dir = "../data/images"  # Root image folder
    query = "persons in hats or caps"
    num_images = 100
    download_images(root_dir, query, num_images)