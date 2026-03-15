import pandas as pd
import re

def clean_shl_data(input_file='data/shl_assessments.csv', 
                   output_file='data/shl_assessments_cleaned.csv'):
    """
    Clean and preprocess the scraped SHL assessment data
    """
    print("\n" + "="*70)
    print("🧹 CLEANING SHL ASSESSMENT DATA")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"\n📊 Loaded {len(df)} assessments")
    
    # 1. Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset='URL')
    duplicates_removed = initial_count - len(df)
    print(f"✅ Removed {duplicates_removed} duplicates")
    
    # 2. Handle missing values
    print(f"\n📋 Missing values:")
    print(f"   - Assessment Name: {df['Assessment Name'].isna().sum()}")
    print(f"   - Short Description: {df['Short Description'].isna().sum()}")
    print(f"   - URL: {df['URL'].isna().sum()}")
    
    # Remove rows with missing critical fields
    df = df.dropna(subset=['Assessment Name', 'URL'])
    
    # Fill missing descriptions
    df['Short Description'] = df['Short Description'].fillna('No description available')
    
    # 3. Clean text fields
    print(f"\n🧼 Cleaning text fields...")
    
    # Clean assessment names
    df['Assessment Name'] = df['Assessment Name'].str.strip()
    df['Assessment Name'] = df['Assessment Name'].str.replace(r'\s+', ' ', regex=True)
    
    # Clean descriptions - remove browser warnings and common noise
    def clean_description(text):
        if pd.isna(text):
            return "No description available"
        
        text = str(text).strip()
        
        # Remove browser compatibility warnings
        noise_patterns = [
            r'If you choose to continue with your current browser.*',
            r'We recommend upgrading to a modern browser.*',
            r'Outdated browser detected.*',
            r'Latest browser options.*',
            r'I understand and wish to continue.*',
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs from text
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # If cleaned text is too short or empty, mark as no description
        if len(text) < 20:
            return "Assessment for evaluating specific skills and competencies"
        
        return text
    
    df['Short Description'] = df['Short Description'].apply(clean_description)
    
    # 4. Create combined text for embeddings
    print(f"🔤 Creating embedding text...")
    
    df['text_for_embedding'] = (
        df['Assessment Name'] + '. ' + df['Short Description']
    )
    
    # Clean combined text
    df['text_for_embedding'] = df['text_for_embedding'].str.replace('\n', ' ')
    df['text_for_embedding'] = df['text_for_embedding'].str.replace(r'\s+', ' ', regex=True)
    df['text_for_embedding'] = df['text_for_embedding'].str.strip()
    
    # 5. Add metadata columns for better retrieval
    print(f"📊 Extracting metadata...")
    
    # Extract assessment type from name (if present)
    def extract_type(name):
        name_lower = name.lower()
        if any(word in name_lower for word in ['personality', 'opq', 'behavior']):
            return 'Personality & Behavior'
        elif any(word in name_lower for word in ['cognitive', 'verify', 'reasoning']):
            return 'Cognitive'
        elif any(word in name_lower for word in ['simulation', 'exercise']):
            return 'Simulation'
        elif any(word in name_lower for word in ['technical', 'programming', 'coding', '.net', 'java', 'python']):
            return 'Technical Skills'
        elif any(word in name_lower for word in ['language', 'writing', 'comprehension']):
            return 'Language'
        else:
            return 'General'
    
    df['assessment_type'] = df['Assessment Name'].apply(extract_type)
    
    # 6. Validate final data
    print(f"\n✅ Validation:")
    print(f"   - Total assessments: {len(df)}")
    print(f"   - Average text length: {df['text_for_embedding'].str.len().mean():.0f} chars")
    print(f"   - Min text length: {df['text_for_embedding'].str.len().min()} chars")
    print(f"   - Max text length: {df['text_for_embedding'].str.len().max()} chars")
    
    # 7. Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"\n💾 SAVED: {output_file}")
    
    # 8. Show samples
    print(f"\n{'='*70}")
    print("📋 SAMPLE CLEANED DATA (first 3):")
    print("="*70)
    
    for i, row in df.head(3).iterrows():
        print(f"\n{i+1}. {row['Assessment Name']}")
        print(f"   Type: {row['assessment_type']}")
        print(f"   Description: {row['Short Description'][:100]}...")
        print(f"   Embedding text: {row['text_for_embedding'][:120]}...")
        print(f"   URL: {row['URL']}")
    
    print(f"\n{'='*70}")
    print("✅ DATA CLEANING COMPLETE")
    print("="*70)
    print(f"\n📊 Assessment Type Distribution:")
    print(df['assessment_type'].value_counts())
    
    return df


def create_data_summary(df):
    """Create a summary report of the cleaned data"""
    summary = {
        'total_assessments': len(df),
        'unique_types': df['assessment_type'].nunique(),
        'avg_name_length': df['Assessment Name'].str.len().mean(),
        'avg_description_length': df['Short Description'].str.len().mean(),
        'avg_embedding_text_length': df['text_for_embedding'].str.len().mean(),
    }
    
    print(f"\n{'='*70}")
    print("📈 DATA SUMMARY REPORT")
    print("="*70)
    for key, value in summary.items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
    
    return summary


def main():
    """Main execution"""
    # Clean the data
    df = clean_shl_data(
        input_file='data/shl_assessments.csv',
        output_file='data/shl_assessments_cleaned.csv'
    )
    
    # Create summary
    create_data_summary(df)
    
    print(f"\n{'='*70}")
    print("🎉 READY FOR NEXT STEPS:")
    print("="*70)
    print("   1. ✅ Data scraped (388 assessments)")
    print("   2. ✅ Data cleaned and preprocessed")
    print("   3. 📊 Next: Generate embeddings")
    print("   4. 🔍 Next: Build vector search")
    print("   5. 🤖 Next: Create recommendation API")
    print("="*70)


if __name__ == "__main__":
    main()