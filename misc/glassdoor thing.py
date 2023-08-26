from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import random
USERNAME = "USERNAME"
PASSWORD = "PASSWORD"
PAGE = 1
# Initialize Chrome WebDriver
driver = webdriver.Chrome()
 
# Log in to Glassdoor
def login_to_glassdoor():
    driver.get("https://www.glassdoor.com/profile/login_input.htm")
    # Wait for page to loa
    time.sleep(2)
    # Find the username and password fields, and input your credentials

    username_field = driver.find_element(By.ID,"inlineUserEmail")
    username_field.send_keys(USERNAME)
    time.sleep(1)
    username_field.submit()
    print("1")
    time.sleep(1)
    # Click the login button
    # login_button = driver.find_element(By.CSS_SELECTOR, "button.email-button")
    # login_button.click()
    print("2")
    time.sleep(1)
    password_field = driver.find_element(By.ID,"inlineUserPassword")
    password_field.send_keys(PASSWORD)
    print("3")
    time.sleep(1)
    login_button = driver.find_element(By.CSS_SELECTOR, "button[data-size-variant='default']")
    login_button.click()
    print("4")
    # Wait for login to complete
    time.sleep(2)

 

# Navigate to the reviews page
def navigate_to_reviews():
    driver.get("https://www.glassdoor.com/Reviews/Honeywell-Reviews-E28_P" + str(PAGE) + ".htm")
    # driver.get("https://www.glassdoor.com/Reviews/Honeywell-Reviews-E28.htm")
    
    # Wait for page to load
    time.sleep(2)
 

# Extract reviews and save to a txt file
def save_reviews_to_txt(a):
    reviews = []
    for _ in range(10):
        try:
            # Find all review elements
            review_elements = driver.find_elements(By.CLASS_NAME,"noBorder.empReview.cf.pb-0.mb-0")
            # pros = driver.find_elements(By.XPATH,'//span[@data-test="pros"]')
            # review_elements = pros
            print("TRY:",a)
            # print(pros)
            for review_element in review_elements:
                try:
                    div_element = driver.find_element(By.CSS_SELECTOR,"div.v2__EIReviewDetailsV2__continueReading.v2__EIReviewDetailsV2__clickable.v2__EIReviewDetailsV2__newUiCta.mb")
                    div_element.click()
                except:
                    pass
                time.sleep(0.1)
                # print("ADDING SHIT")
                # print(review_element)
                review_text = review_element.text
                reviews.append(review_text)
                time.sleep(0.05)
                # print(review_text)
                # break
            break
        except:
            continue

    # Save the reviews to a txt file

    with open(str(a) + "glassdoor_reviews.txt", "w", encoding="utf-8") as file:
        file.write("\n\n".join(reviews))
    print("Reviews saved to glassdoor_reviews.txt")

 

def next_page():
    max_retries = 20
    for a in range(max_retries):
        try:
            button = driver.find_element(By.CSS_SELECTOR,"button[aria-label='Next']")
            button.click()
            time.sleep(0.2)
            return
        except:
            try:
                button = driver.find_element(By.CLASS_NAME,"qual_x_close")
                button.click()
            except:
                pass
            continue
    raise ValueError('no next page code stupid')
# Execute the entire process
def main():
    login_to_glassdoor()
    navigate_to_reviews()
    for a in range(PAGE,1000000000):
        save_reviews_to_txt(a)
        next_page()
        time.sleep(1)
    driver.quit()
 
if __name__ == "__main__":
    main()