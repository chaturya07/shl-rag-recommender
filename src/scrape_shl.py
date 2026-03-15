
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urljoin

class SHLScraperFinal:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.assessments = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.shl.com/',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_assessment_links_from_page(self, page_num=1):
        """Get assessment links from a single catalog page"""
        if page_num == 1:
            url = f"{self.base_url}/products/product-catalog/?type=1"
        else:
            start = (page_num - 1) * 12
            url = f"{self.base_url}/products/product-catalog/?start={start}&type=1"
        
        try:
            print(f"  Fetching page {page_num}: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links to individual assessment pages
            links = set()
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '/products/product-catalog/view/' in href:
                    full_url = urljoin(self.base_url, href)
                    clean_url = full_url.split('#')[0].split('?')[0]
                    links.add(clean_url)
            
            return list(links)
            
        except Exception as e:
            print(f"  ❌ Error on page {page_num}: {e}")
            return []
    
    def get_all_assessment_links(self):
        """Get all assessment links by paginating through catalog"""
        all_links = set()
        page_num = 1
        max_pages = 35  # Safety limit (377 assessments / 12 per page ≈ 32 pages)
        
        print("\n" + "="*70)
        print("🔍 STEP 1: EXTRACTING ASSESSMENT LINKS FROM CATALOG")
        print("="*70)
        
        while page_num <= max_pages:
            links = self.get_assessment_links_from_page(page_num)
            
            if links:
                new_links = [l for l in links if l not in all_links]
                all_links.update(links)
                print(f"  ✓ Found {len(new_links)} new links | Total: {len(all_links)}")
                
                time.sleep(random.uniform(1, 2))
                page_num += 1
            else:
                print(f"  ✓ No more links found")
                break
        
        print(f"\n{'='*70}")
        print(f"📊 TOTAL UNIQUE LINKS: {len(all_links)}")
        print(f"🎯 TARGET: 377 assessments")
        print(f"{'='*70}")
        
        return list(all_links)
    
    def scrape_assessment_page(self, url):
        """Scrape individual assessment page for name, description, URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract name from h1 or title
            name = None
            h1 = soup.find('h1')
            if h1:
                name = h1.get_text(strip=True)
            
            if not name:
                title = soup.find('title')
                if title:
                    name = title.get_text(strip=True).replace(' | SHL', '').strip()
            
            # Extract description from first meaningful paragraph
            description = ""
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if (len(text) > 50 and 
                    'cookie' not in text.lower() and 
                    'privacy' not in text.lower() and
                    'speak to' not in text.lower()):
                    description = text
                    break
            
            # Limit to ~200 characters (2-3 lines)
            if len(description) > 200:
                description = description[:200].rsplit(' ', 1)[0] + "..."
            
            if not description:
                description = "No description available"
            
            return {
                'Assessment Name': name or 'Unknown Assessment',
                'Short Description': description,
                'URL': url
            }
            
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:50]}")
            return None
    
    def scrape_all(self):
        """Main scraping function"""
        print("\n🚀 SHL ASSESSMENT SCRAPER")
        print("="*70)
        
        # Get all assessment links
        links = self.get_all_assessment_links()
        
        if not links:
            print("\n❌ No assessment links found!")
            return
        
        # Scrape each assessment page
        print(f"\n{'='*70}")
        print("🔄 STEP 2: SCRAPING INDIVIDUAL ASSESSMENT PAGES")
        print(f"{'='*70}")
        
        for i, url in enumerate(links, 1):
            assessment_name = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
            print(f"\n[{i}/{len(links)}] {assessment_name}")
            
            data = self.scrape_assessment_page(url)
            
            if data:
                self.assessments.append(data)
                print(f"  ✓ {data['Assessment Name'][:60]}")
            
            # Respectful delay
            time.sleep(random.uniform(2, 4))
            
            # Progress checkpoint
            if i % 50 == 0:
                print(f"\n{'='*70}")
                print(f"📊 PROGRESS: {i}/{len(links)} ({i/len(links)*100:.1f}%)")
                print(f"{'='*70}")
        
        print(f"\n{'='*70}")
        print(f"✅ SCRAPING COMPLETE")
        print(f"{'='*70}")
    
    def save_to_csv(self, filename='data/shl_assessments.csv'):
        """Save assessments to CSV"""
        if not self.assessments:
            print("\n❌ No assessments to save")
            return
        
        df = pd.DataFrame(self.assessments)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\n💾 SAVED TO: {filename}")
        print(f"\n{'='*70}")
        print("📈 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Total assessments: {len(self.assessments)}")
        print(f"🎯 Target: 377")
        print(f"📊 Coverage: {len(self.assessments)/377*100:.1f}%")
        
        # Show sample
        print(f"\n📋 SAMPLE (first 5):")
        print("-"*70)
        for i, assessment in enumerate(self.assessments[:5], 1):
            print(f"\n{i}. {assessment['Assessment Name']}")
            print(f"   {assessment['Short Description'][:80]}...")
            print(f"   {assessment['URL']}")
        
        print(f"\n{'='*70}")
        print("🎉 SUCCESS! Ready for next steps:")
        print("   1. ✅ Assessment data collected")
        print("   2. 📊 Build embeddings (Sentence-BERT/OpenAI)")
        print("   3. 🔍 Create vector search (FAISS/Pinecone)")
        print("   4. 🤖 Build recommendation API")
        print("   5. 🌐 Deploy frontend + API")
        print(f"{'='*70}")


def main():
    """Run the scraper"""
    scraper = SHLScraperFinal()
    
    try:
        scraper.scrape_all()
        scraper.save_to_csv()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if scraper.assessments:
            print("💾 Saving partial results...")
            scraper.save_to_csv('data/shl_assessments_partial.csv')
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if scraper.assessments:
            print("💾 Saving partial results...")
            scraper.save_to_csv('data/shl_assessments_partial.csv')


if __name__ == "__main__":
    main()


# ==========================================
# QUICK TEST FUNCTION (test with 10 only)
# ==========================================
def quick_test():
    """Test scraper on first 10 assessments"""
    print("🧪 QUICK TEST MODE (10 assessments)")
    scraper = SHLScraperFinal()
    
    # Get first page of links only
    links = scraper.get_assessment_links_from_page(1)
    
    if links:
        print(f"\n✓ Testing with {min(10, len(links))} assessments...")
        
        for url in links[:10]:
            print(f"\n🔍 {url.split('/')[-2]}")
            data = scraper.scrape_assessment_page(url)
            if data:
                scraper.assessments.append(data)
                print(f"  ✓ {data['Assessment Name']}")
            time.sleep(1)
        
        scraper.save_to_csv('data/shl_test.csv')
        print("\n✅ Test complete! Check data/shl_test.csv")
