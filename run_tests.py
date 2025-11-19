import csv
import argparse
import os
import sys
import json
from dotenv import load_dotenv
import llm
from schema import ExtractionResult

# Load environment variables
load_dotenv()

def load_test_cases(csv_path):
    cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cases.append(row)
    return cases

def run_test(case_id, cases):
    case = next((c for c in cases if c['id'] == str(case_id)), None)
    if not case:
        print(f"Test case {case_id} not found.")
        return

    print(f"--- Running Test Case {case_id} ---")
    print(f"Category: {case['category']}")
    print(f"Message: {case['message']}")
    print(f"Expected: {case['expected']}")
    
    if not case.get('message'):
        print(f"Skipping case {case_id}: No message found.")
        return
    message = case['message'].strip('"') # Remove surrounding quotes if present in CSV parsing
    
    try:
        result = llm.extract_tasks(message)
        print("\nActual Result (JSON):")
        print(result.model_dump_json(indent=2))
        
        # Basic Verification
        print("\n--- Verification ---")
        if case['category'] == 'create':
            if result.tasks_new:
                print("✅ tasks_new is not empty")
                for t in result.tasks_new:
                    print(f"   - Title: {t.title}")
                    print(f"   - Project: {t.project}")
                    print(f"   - Deadline: {t.deadline}")
            else:
                print("❌ tasks_new is empty")
        elif case['category'] == 'clarify':
             if result.clarifying_questions:
                 print(f"✅ Clarifying questions asked: {len(result.clarifying_questions)}")
             else:
                 print("❌ No clarifying questions asked")
        
    except Exception as e:
        print(f"❌ Error running test: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run Telegram Bot Test Cases')
    parser.add_argument('--id', type=int, help='Test case ID to run')
    parser.add_argument('--all', action='store_true', help='Run all test cases (be careful with API usage)')
    
    args = parser.parse_args()
    
    csv_path = 'test_cases.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    cases = load_test_cases(csv_path)

    if args.id:
        run_test(args.id, cases)
    elif args.all:
        for case in cases:
            run_test(case['id'], cases)
            print("\n" + "="*30 + "\n")
    else:
        print("Please specify --id <ID> or --all")

if __name__ == "__main__":
    main()
