# Evaluation Questions for LSTM News QA System

## Test Questions for System Evaluation

### Definition Questions (What)
1. What is artificial intelligence?
2. What is quantum computing?
3. What is machine learning?
4. What is open source software?
5. What is cybersecurity?

### Person Questions (Who)
6. Who developed OpenAI?
7. Who is mentioned in the AI articles?
8. Who are the researchers working on quantum computing?
9. Who founded the companies mentioned?
10. Who wrote the articles about technology?

### Time Questions (When)
11. When was the latest AI model released?
12. When did the cybersecurity incident happen?
13. When was the research published?
14. When did the company announce the product?
15. When was the technology developed?

### Location Questions (Where)
16. Where is the research being conducted?
17. Where are the companies located?
18. Where did the events take place?
19. Where was the technology developed?
20. Where are the security vulnerabilities found?

### Explanation Questions (How/Why)
21. How does machine learning work?
22. Why is cybersecurity important?
23. How do quantum computers function?
24. Why are companies investing in AI?
25. How does open source development work?

### Complex Questions
26. What are the main security risks mentioned in the articles?
27. What companies are developing artificial intelligence technology?
28. What are the recent developments in quantum computing research?
29. What are the advantages of open source software development?
30. What challenges do AI companies face?

### Questions Likely to Trigger Fallback
31. What is the meaning of life?
32. Who invented the wheel?
33. When will humans colonize Mars?
34. Where is Atlantis located?
35. How do you make a perfect pizza?

### Domain-Specific Questions
36. What security vulnerabilities were discovered recently?
37. Which AI models were released by OpenAI?
38. What are the applications of quantum sensors?
39. What open source projects are mentioned?
40. What are the latest developments in robotics?

## Expected Behavior

### High Confidence Answers (Should Return Answer)
- Questions 1-30 should generally return confident answers with proper citations
- Answers should include relevant information from the news articles
- Citations should include URL, headline, date, and publisher

### Fallback Responses (Should Return "not found")
- Questions 31-35 should trigger fallback responses
- System should provide top related sources even when no confident answer is found
- Confidence scores should be below the threshold (default 0.3)

### Quality Metrics
- **Exact Match (EM)**: Percentage of questions with exactly correct answers
- **F1 Score**: Token-level overlap between predicted and correct answers
- **Citation Accuracy**: Percentage of answers with correct source attribution
- **Confidence Calibration**: Correlation between confidence scores and answer quality

## Usage

Save questions to a file (one per line) and run:
```bash
python Week4/inference_pipeline.py --batch_file evaluation_questions.txt --output_file results.json
```

Or test individual questions:
```bash
python demo.py "What is artificial intelligence?"
```