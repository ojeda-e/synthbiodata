// Initialize Mermaid when the page loads
document.addEventListener('DOMContentLoaded', function() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            themeVariables: {
                primaryColor: '#7D5699',
                primaryTextColor: '#3B464F',
                primaryBorderColor: '#D0D1E7',
                lineColor: '#3B464F',
                secondaryColor: '#F7C5C5',
                tertiaryColor: '#D0D1E7'
            }
        });
    }
});

