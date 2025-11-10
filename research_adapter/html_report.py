"""
HTML report generation for grammar analysis results
"""

from typing import Dict, Any, List
from .config import REPORT_CONFIG, ISSUE_WEIGHTS, GEC_CONFIG


class HTMLReportGenerator:
    """Generate HTML reports from grammar analysis results"""
    
    def __init__(self):
        self.color_scheme = REPORT_CONFIG['color_scheme']
    
    def _get_grade_info(self, score: float) -> Dict[str, Any]:
        """Get grade information based on score"""
        if score >= 85:
            return {
                'grade': 'A',
                'message': '🌟 Excellent work! Your grammar is really strong!',
                'color': self.color_scheme['excellent']
            }
        elif score >= 75:
            return {
                'grade': 'B',
                'message': "👍 Good job! You're doing well with just a few areas to polish.",
                'color': self.color_scheme['good']
            }
        elif score >= 65:
            return {
                'grade': 'C',
                'message': "📚 You're on the right track! A bit more practice and you'll be great.",
                'color': self.color_scheme['average']
            }
        elif score >= 55:
            return {
                'grade': 'D',
                'message': "💪 Keep practicing! Every mistake is a learning opportunity.",
                'color': self.color_scheme['needs_work']
            }
        else:
            return {
                'grade': 'F',
                'message': "🌱 Don't worry! Everyone starts somewhere. Let's work on this together!",
                'color': self.color_scheme['needs_work']
            }
    
    def _get_severity_style(self, severity: str) -> str:
        """Get CSS style for severity level"""
        severity_colors = {
            'Critical': self.color_scheme['critical'],
            'Major': self.color_scheme['major'],
            'Minor': self.color_scheme['minor'],
            'Suggestion': self.color_scheme['suggestion']
        }
        color = severity_colors.get(severity, '#6c757d')
        return f"color: {color}; font-weight: bold;"
    
    def _generate_issues_section(self, issues: List[Dict[str, Any]]) -> str:
        """Generate HTML for issues section"""
        if not issues:
            return """
            <div class="alert alert-success">
                <h4>🎉 Fantastic news!</h4>
                <p>I didn't find any issues to address. Your grammar looks great! 🌟</p>
            </div>
            """
        
        # Group issues by severity
        from collections import defaultdict
        severity_groups = defaultdict(list)
        for issue in issues:
            severity_groups[issue['severity']].append(issue)
        
        html = '<div class="issues-section">'
        
        severity_icons = {
            'Critical': '🚨',
            'Major': '⚠️',
            'Minor': '💡',
            'Suggestion': '💭'
        }
        
        for severity in ['Critical', 'Major', 'Minor', 'Suggestion']:
            if severity in severity_groups:
                html += f'''
                <div class="severity-group">
                    <h3 style="{self._get_severity_style(severity)}">
                        {severity_icons[severity]} {severity} Areas to Consider
                    </h3>
                    <ul class="issue-list">
                '''
                
                for i, issue in enumerate(severity_groups[severity], 1):
                    html += f'''
                    <li class="issue-item">
                        <div class="issue-header">
                            <strong>{i}. {issue['type']}</strong>
                        </div>
                        <div class="issue-context">
                            Context: <em>"{issue['context']}"</em>
                        </div>
                        <div class="issue-correction">
                            💬 {issue['correction']}
                        </div>
                    </li>
                    '''
                
                html += '</ul></div>'
        
        html += '</div>'
        return html
    
    def _generate_category_counts_section(self, category_counts: Dict[str, int]) -> str:
        """Generate HTML for category counts section"""
        if not category_counts:
            return ""
        
        html = '''
        <div class="category-section" style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3>📊 Issue Categories</h3>
            <div class="category-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
        '''
        
        # Sort categories by count (descending)
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            html += f'''
                <div class="category-item" style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                    <div style="font-weight: bold; color: #2c3e50; margin-bottom: 5px;">{category}</div>
                    <div style="font-size: 1.5em; color: #007bff; font-weight: bold;">{count}</div>
                </div>
            '''
        
        html += '''
            </div>
        </div>
        '''
        
        return html
    
    def _generate_complexity_section(self, complexity_stats: Dict[str, Any]) -> str:
        """Generate HTML for complexity statistics section"""
        if not complexity_stats or complexity_stats.get('complexity_score', 0) == 0:
            return ""
        
        complexity_score = complexity_stats.get('complexity_score', 0)
        avg_sentence_length = complexity_stats.get('avg_sentence_length', 0)
        avg_word_length = complexity_stats.get('avg_word_length', 0)
        dependency_depth = complexity_stats.get('dependency_depth', 0)
        subordinate_clauses = complexity_stats.get('subordinate_clauses', 0)
        noun_phrases = complexity_stats.get('noun_phrases', 0)
        
        # Determine complexity level
        if complexity_score < 0.3:
            complexity_level = "Simple"
            complexity_color = "#28a745"
        elif complexity_score < 0.6:
            complexity_level = "Moderate"
            complexity_color = "#ffc107"
        else:
            complexity_level = "Complex"
            complexity_color = "#dc3545"
        
        html = f'''
        <div class="complexity-section" style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3>📈 Text Complexity Analysis</h3>
            <div class="complexity-stats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-top: 15px;">
                <div class="complexity-item" style="background: white; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">Complexity Level</div>
                    <div style="font-size: 1.3em; color: {complexity_color}; font-weight: bold;">{complexity_level}</div>
                    <div style="font-size: 0.8em; color: #6c757d;">({complexity_score:.2f})</div>
                </div>
                <div class="complexity-item" style="background: white; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">Avg Sentence Length</div>
                    <div style="font-size: 1.3em; color: #2c3e50; font-weight: bold;">{avg_sentence_length:.1f} words</div>
                </div>
                <div class="complexity-item" style="background: white; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">Avg Word Length</div>
                    <div style="font-size: 1.3em; color: #2c3e50; font-weight: bold;">{avg_word_length:.1f} chars</div>
                </div>
                <div class="complexity-item" style="background: white; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">Dependency Depth</div>
                    <div style="font-size: 1.3em; color: #2c3e50; font-weight: bold;">{dependency_depth:.1f}</div>
                </div>
                <div class="complexity-item" style="background: white; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">Subordinate Clauses</div>
                    <div style="font-size: 1.3em; color: #2c3e50; font-weight: bold;">{subordinate_clauses}</div>
                </div>
                <div class="complexity-item" style="background: white; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">Noun Phrases</div>
                    <div style="font-size: 1.3em; color: #2c3e50; font-weight: bold;">{noun_phrases}</div>
                </div>
            </div>
        </div>
        '''
        
        return html
    
    def _generate_summary_section(self, analysis_result: Dict[str, Any]) -> str:
        """Generate HTML for summary section"""
        score = analysis_result['final_score']
        word_count = analysis_result['word_count']
        sentence_count = analysis_result['sentence_count']
        issue_count = len(analysis_result['all_issues'])
        rule_score = analysis_result.get('rule_score', score)
        languagetool_matches = analysis_result.get('languagetool_matches', 0)
        score_gec = analysis_result.get('score_gec', 100.0)
        
        grade_info = self._get_grade_info(score)
        
        html = f'''
        <div class="summary-section" style="background: linear-gradient(135deg, {grade_info['color']}15 0%, {grade_info['color']}05 100%); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
            <h2>📊 Your Grammar Assessment Summary</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-label">📝 Words analyzed:</span>
                    <span class="stat-value">{word_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">📄 Sentences analyzed:</span>
                    <span class="stat-value">{sentence_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">🔍 Total suggestions:</span>
                    <span class="stat-value">{issue_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">📈 Your score:</span>
                    <span class="stat-value" style="font-size: 1.5em; color: {grade_info['color']}; font-weight: bold;">{score:.1f}/100</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">🏆 Grade:</span>
                    <span class="stat-value" style="font-size: 1.5em; color: {grade_info['color']}; font-weight: bold;">{grade_info['grade']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">📋 Rule-based score:</span>
                    <span class="stat-value">{rule_score:.1f}/100</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">🔧 LanguageTool matches:</span>
                    <span class="stat-value">{languagetool_matches}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">✏️ GEC score:</span>
                    <span class="stat-value">{score_gec:.1f}/100</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">🤖 Text-only score:</span>
                    <span class="stat-value">{analysis_result.get('score_text_only', 75.0):.1f}/100</span>
                </div>
            </div>
            <div class="grade-message" style="margin-top: 20px; padding: 15px; background: white; border-radius: 5px; border-left: 4px solid {grade_info['color']};">
                <p style="margin: 0; font-size: 1.1em;">{grade_info['message']}</p>
            </div>
        </div>
        '''
        
        return html
    
    def _generate_gec_split_view(self, gec_results: Dict[str, Any]) -> str:
        """Generate HTML for Original/Corrected split view"""
        if not gec_results:
            return ""
        
        original = gec_results.get('original', '')
        corrected = gec_results.get('corrected', '')
        edit_metrics = gec_results.get('edit_metrics', {})
        score_gec = gec_results.get('score_gec', 100.0)
        
        # Determine if there are changes
        has_changes = edit_metrics.get('has_changes', False)
        
        # Format text with line breaks for readability
        def format_text(text):
            # Split into sentences for better display
            import re
            sentences = re.split(r'([.!?]+)', text)
            formatted = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    formatted.append(sentences[i] + sentences[i+1])
            return '<br>'.join(formatted) if formatted else text
        
        original_formatted = format_text(original)
        corrected_formatted = format_text(corrected)
        
        # Get score color
        if score_gec >= 85:
            score_color = "#28a745"
        elif score_gec >= 70:
            score_color = "#20c997"
        elif score_gec >= 55:
            score_color = "#ffc107"
        else:
            score_color = "#dc3545"
        
        html = f'''
        <div class="gec-section" style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h2>✏️ Grammatical Error Correction (GEC)</h2>
            <div style="margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="margin: 0; color: #2c3e50;">GEC Score</h3>
                    <div style="font-size: 2em; font-weight: bold; color: {score_color};">
                        {score_gec:.1f}/100
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 15px;">
                    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <div style="font-size: 0.9em; color: #6c757d;">Edits</div>
                        <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">{edit_metrics.get('num_edits', 0)}</div>
                    </div>
                    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <div style="font-size: 0.9em; color: #6c757d;">Edit Rate</div>
                        <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">{edit_metrics.get('edit_rate', 0.0):.2%}</div>
                    </div>
                    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <div style="font-size: 0.9em; color: #6c757d;">Precision</div>
                        <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">{edit_metrics.get('precision', 1.0):.2f}</div>
                    </div>
                    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <div style="font-size: 0.9em; color: #6c757d;">Recall</div>
                        <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">{edit_metrics.get('recall', 1.0):.2f}</div>
                    </div>
                    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <div style="font-size: 0.9em; color: #6c757d;">F1 Score</div>
                        <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">{edit_metrics.get('f1', 1.0):.2f}</div>
                    </div>
                </div>
            </div>
            
            <div class="split-view" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                <div class="original-panel" style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #dc3545;">
                    <h3 style="margin-top: 0; color: #dc3545; display: flex; align-items: center;">
                        <span style="margin-right: 10px;">📝</span> Original Text
                    </h3>
                    <div class="original-text" style="color: #495057; line-height: 1.8; font-size: 1.05em; padding: 15px; background: #f8f9fa; border-radius: 5px; min-height: 100px;">
                        {original_formatted}
                    </div>
                </div>
                
                <div class="corrected-panel" style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h3 style="margin-top: 0; color: #28a745; display: flex; align-items: center;">
                        <span style="margin-right: 10px;">✅</span> Corrected Text
                    </h3>
                    <div class="corrected-text" style="color: #495057; line-height: 1.8; font-size: 1.05em; padding: 15px; background: #f8f9fa; border-radius: 5px; min-height: 100px;">
                        {corrected_formatted}
                    </div>
                </div>
            </div>
            
            {f'<div style="margin-top: 15px; padding: 10px; background: #d4edda; border-left: 4px solid #28a745; border-radius: 5px; color: #155724;"><strong>ℹ️</strong> {edit_metrics.get("num_edits", 0)} edit(s) were made to improve the grammar.</div>' if has_changes else '<div style="margin-top: 15px; padding: 10px; background: #d1ecf1; border-left: 4px solid #17a2b8; border-radius: 5px; color: #0c5460;"><strong>ℹ️</strong> No corrections were needed. Your text is grammatically correct!</div>'}
        </div>
        
        <style>
            @media (max-width: 768px) {{
                .split-view {{
                    grid-template-columns: 1fr !important;
                }}
            }}
        </style>
        '''
        
        return html
    
    def generate_html(self, analysis_result: Dict[str, Any], transcript: str) -> str:
        """Generate complete HTML report"""
        issues_html = self._generate_issues_section(analysis_result['all_issues'])
        summary_html = self._generate_summary_section(analysis_result)
        
        # Generate category counts section
        category_counts = analysis_result.get('category_counts', {})
        category_html = self._generate_category_counts_section(category_counts)
        
        # Generate complexity section
        complexity_stats = analysis_result.get('complexity_stats', {})
        complexity_html = self._generate_complexity_section(complexity_stats)
        
        # Generate GEC split view
        gec_results = analysis_result.get('gec_results', {})
        gec_html = self._generate_gec_split_view(gec_results) if REPORT_CONFIG.get('include_gec_view', True) else ""
        
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grammar Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        h3 {{
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .transcript-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #007bff;
        }}
        
        .transcript-text {{
            font-size: 1.1em;
            line-height: 1.8;
            color: #495057;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .stat-label {{
            display: block;
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            display: block;
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .issues-section {{
            margin-top: 30px;
        }}
        
        .severity-group {{
            margin-bottom: 30px;
        }}
        
        .issue-list {{
            list-style: none;
            padding: 0;
        }}
        
        .issue-item {{
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #6c757d;
        }}
        
        .issue-header {{
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .issue-context {{
            color: #6c757d;
            font-style: italic;
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
        
        .issue-correction {{
            color: #495057;
            padding: 10px;
            background: #e7f3ff;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        }}
        
        .alert {{
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .alert-success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .summary-stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Grammar Analysis Report</h1>
        <p style="color: #6c757d; margin-bottom: 30px;">Comprehensive analysis of your spoken English</p>
        
        <div class="transcript-section">
            <h3>📝 Transcript</h3>
            <div class="transcript-text">"{transcript}"</div>
        </div>
        
        {summary_html}
        
        {category_html}
        
        {complexity_html}
        
        {gec_html}
        
        <div class="issues-section">
            <h2>📋 Detailed Analysis</h2>
            {issues_html}
        </div>
        
        <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center;">
            <p style="color: #6c757d; margin: 0;">
                💡 Remember: Every great writer started somewhere!<br>
                Keep practicing, and you'll keep improving. You've got this! 💪
            </p>
        </div>
    </div>
</body>
</html>
        '''
        
        return html

