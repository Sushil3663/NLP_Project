"""
Citation formatting utilities for the QA system.
Formats answers with proper citations including URL, headline, date, and publisher.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re

class CitationFormatter:
    """Formats citations and final answers for the QA system."""
    
    def __init__(self):
        self.citation_styles = {
            'standard': self._format_standard_citation,
            'brief': self._format_brief_citation,
            'academic': self._format_academic_citation,
            'news': self._format_news_citation
        }
    
    def clean_title(self, title: str) -> str:
        """Clean and format article title."""
        if not title:
            return "Untitled Article"
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Ensure title ends with appropriate punctuation
        if not title.endswith(('.', '!', '?')):
            title += '.'
        
        return title
    
    def clean_publisher(self, publisher: str) -> str:
        """Clean and format publisher name."""
        if not publisher:
            return "Unknown Publisher"
        
        # Remove common suffixes
        publisher = re.sub(r'\.(com|org|net|edu)$', '', publisher, flags=re.IGNORECASE)
        
        # Capitalize properly
        publisher = publisher.title()
        
        return publisher
    
    def format_date(self, date_str: str) -> str:
        """Format date string for citation."""
        if not date_str:
            return "No date"
        
        try:
            # Try to parse ISO format
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime("%B %d, %Y")
            
            # Try other common formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    dt = datetime.strptime(date_str[:10], fmt)
                    return dt.strftime("%B %d, %Y")
                except ValueError:
                    continue
            
            # If parsing fails, return as-is
            return date_str[:10] if len(date_str) >= 10 else date_str
            
        except Exception:
            return date_str if date_str else "No date"
    
    def _format_standard_citation(self, passage_info: Dict[str, Any]) -> str:
        """Format standard citation style."""
        title = self.clean_title(passage_info.get('title', ''))
        publisher = self.clean_publisher(passage_info.get('publisher', ''))
        date = self.format_date(passage_info.get('date', ''))
        url = passage_info.get('url', '')
        
        citation = f"{title} {publisher}, {date}."
        
        if url:
            citation += f" Available at: {url}"
        
        return citation
    
    def _format_brief_citation(self, passage_info: Dict[str, Any]) -> str:
        """Format brief citation style."""
        publisher = self.clean_publisher(passage_info.get('publisher', ''))
        date = self.format_date(passage_info.get('date', ''))
        
        return f"({publisher}, {date})"
    
    def _format_academic_citation(self, passage_info: Dict[str, Any]) -> str:
        """Format academic citation style."""
        title = self.clean_title(passage_info.get('title', ''))
        publisher = self.clean_publisher(passage_info.get('publisher', ''))
        date = self.format_date(passage_info.get('date', ''))
        url = passage_info.get('url', '')
        author = passage_info.get('author', '')
        
        citation_parts = []
        
        if author:
            citation_parts.append(f"{author}.")
        
        citation_parts.append(f'"{title}"')
        citation_parts.append(f"{publisher},")
        citation_parts.append(f"{date}.")
        
        if url:
            citation_parts.append(f"Web. {url}")
        
        return " ".join(citation_parts)
    
    def _format_news_citation(self, passage_info: Dict[str, Any]) -> str:
        """Format news-style citation."""
        title = self.clean_title(passage_info.get('title', ''))
        publisher = self.clean_publisher(passage_info.get('publisher', ''))
        date = self.format_date(passage_info.get('date', ''))
        
        return f"{title} - {publisher} ({date})"
    
    def format_citation(self, passage_info: Dict[str, Any], style: str = 'standard') -> str:
        """Format citation using specified style."""
        if style not in self.citation_styles:
            style = 'standard'
        
        return self.citation_styles[style](passage_info)
    
    def format_multiple_citations(self, 
                                 passages: List[Dict[str, Any]], 
                                 style: str = 'standard',
                                 max_citations: int = 3) -> List[str]:
        """Format multiple citations."""
        citations = []
        
        for passage in passages[:max_citations]:
            citation = self.format_citation(passage, style)
            citations.append(citation)
        
        return citations
    
    def format_final_answer(self, 
                           answer_data: Dict[str, Any],
                           citation_style: str = 'standard',
                           include_confidence: bool = True) -> Dict[str, Any]:
        """
        Format the final answer with citations.
        
        Args:
            answer_data: Dictionary containing answer and supporting passages
            citation_style: Style for citations
            include_confidence: Whether to include confidence score
        
        Returns:
            Formatted answer dictionary
        """
        if answer_data.get('is_fallback', False):
            return self._format_fallback_answer(answer_data)
        
        answer_text = answer_data.get('answer_text', '')
        confidence = answer_data.get('confidence', 0.0)
        supporting_passages = answer_data.get('supporting_passages', [])
        
        # Format main citation (from primary source)
        primary_citation = ""
        if supporting_passages:
            primary_source = supporting_passages[0]
            primary_citation = self.format_citation(primary_source, citation_style)
        
        # Format additional citations if multiple sources
        additional_citations = []
        if len(supporting_passages) > 1:
            additional_citations = self.format_multiple_citations(
                supporting_passages[1:], citation_style, max_citations=2
            )
        
        # Build formatted response
        formatted_response = {
            'answer': answer_text,
            'primary_citation': primary_citation,
            'additional_citations': additional_citations,
            'source_count': len(supporting_passages),
            'answer_type': answer_data.get('answer_type', 'unknown')
        }
        
        if include_confidence:
            formatted_response['confidence'] = confidence
            formatted_response['confidence_level'] = self._get_confidence_level(confidence)
        
        # Add source details
        if supporting_passages:
            primary_source = supporting_passages[0]
            formatted_response['source_details'] = {
                'title': primary_source.get('title', ''),
                'url': primary_source.get('url', ''),
                'publisher': self.clean_publisher(primary_source.get('publisher', '')),
                'date': self.format_date(primary_source.get('date', '')),
                'author': primary_source.get('author', '')
            }
        
        return formatted_response
    
    def _format_fallback_answer(self, fallback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format fallback response when no confident answer is found."""
        top_sources = fallback_data.get('top_sources', [])
        
        # Format top sources for reference
        formatted_sources = []
        for source in top_sources:
            formatted_source = {
                'title': self.clean_title(source.get('title', '')),
                'publisher': self.clean_publisher(source.get('publisher', '')),
                'date': self.format_date(source.get('date', '')),
                'url': source.get('url', ''),
                'relevance_score': source.get('relevance_score', 0.0)
            }
            formatted_sources.append(formatted_source)
        
        return {
            'answer': 'not found',
            'message': fallback_data.get('message', 'No confident answer found.'),
            'confidence': 0.0,
            'confidence_level': 'none',
            'is_fallback': True,
            'suggested_sources': formatted_sources,
            'source_count': len(formatted_sources)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to descriptive level."""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.4:
            return 'low'
        else:
            return 'very low'
    
    def format_display_text(self, formatted_answer: Dict[str, Any]) -> str:
        """Format answer for display/printing."""
        if formatted_answer.get('is_fallback', False):
            return self._format_fallback_display(formatted_answer)
        
        lines = []
        
        # Answer
        lines.append(f"Answer: {formatted_answer['answer']}")
        
        # Confidence
        if 'confidence' in formatted_answer:
            confidence = formatted_answer['confidence']
            level = formatted_answer.get('confidence_level', '')
            lines.append(f"Confidence: {confidence:.2f} ({level})")
        
        # Primary citation
        if formatted_answer.get('primary_citation'):
            lines.append(f"Source: {formatted_answer['primary_citation']}")
        
        # Additional sources
        additional_citations = formatted_answer.get('additional_citations', [])
        if additional_citations:
            lines.append("Additional sources:")
            for i, citation in enumerate(additional_citations, 1):
                lines.append(f"  {i}. {citation}")
        
        # Source count
        source_count = formatted_answer.get('source_count', 0)
        if source_count > 1:
            lines.append(f"Based on {source_count} sources")
        
        return "\n".join(lines)
    
    def _format_fallback_display(self, fallback_answer: Dict[str, Any]) -> str:
        """Format fallback answer for display."""
        lines = []
        
        lines.append(f"Answer: {fallback_answer['answer']}")
        lines.append(f"Message: {fallback_answer['message']}")
        
        suggested_sources = fallback_answer.get('suggested_sources', [])
        if suggested_sources:
            lines.append("\nTop related sources:")
            for i, source in enumerate(suggested_sources, 1):
                title = source['title']
                publisher = source['publisher']
                date = source['date']
                lines.append(f"  {i}. {title} - {publisher} ({date})")
                if source.get('url'):
                    lines.append(f"     {source['url']}")
        
        return "\n".join(lines)

if __name__ == "__main__":
    # Test citation formatting
    formatter = CitationFormatter()
    
    # Sample passage info
    passage_info = {
        'title': 'Artificial Intelligence Breakthrough in 2024',
        'publisher': 'techcrunch.com',
        'date': '2024-01-15T10:30:00-05:00',
        'url': 'https://techcrunch.com/ai-breakthrough',
        'author': 'John Smith'
    }
    
    print("Citation Formatting Test:")
    print("=" * 30)
    
    # Test different citation styles
    styles = ['standard', 'brief', 'academic', 'news']
    
    for style in styles:
        citation = formatter.format_citation(passage_info, style)
        print(f"\n{style.title()} style:")
        print(citation)
    
    # Test full answer formatting
    print("\n" + "=" * 30)
    print("Full Answer Formatting Test:")
    
    answer_data = {
        'answer_text': 'Artificial intelligence is intelligence demonstrated by machines.',
        'confidence': 0.85,
        'answer_type': 'definition',
        'supporting_passages': [
            passage_info,
            {
                'title': 'Understanding AI Technology',
                'publisher': 'wired.com',
                'date': '2024-01-10',
                'url': 'https://wired.com/ai-tech'
            }
        ]
    }
    
    formatted = formatter.format_final_answer(answer_data)
    display_text = formatter.format_display_text(formatted)
    
    print("\nFormatted Answer:")
    print(display_text)
    
    # Test fallback formatting
    print("\n" + "=" * 30)
    print("Fallback Answer Test:")
    
    fallback_data = {
        'is_fallback': True,
        'message': 'Could not find a confident answer to your question.',
        'top_sources': [
            {
                'title': 'Related Article 1',
                'publisher': 'example.com',
                'date': '2024-01-01',
                'url': 'https://example.com/1',
                'relevance_score': 0.7
            }
        ]
    }
    
    fallback_formatted = formatter.format_final_answer(fallback_data)
    fallback_display = formatter.format_display_text(fallback_formatted)
    
    print("\nFallback Response:")
    print(fallback_display)