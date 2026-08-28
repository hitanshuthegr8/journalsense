import os

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Scholarly Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .stat-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to fetch data from OpenAlex API
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_openalex_data(endpoint, params=None):
    """
    Fetch data from OpenAlex API
    
    Parameters:
    endpoint (str): API endpoint (works, authors, venues, institutions, concepts)
    params (dict): Query parameters
    
    Returns:
    dict: JSON response
    """
    base_url = "https://api.openalex.org"
    url = f"{base_url}/{endpoint}"
    
    # OpenAlex "polite pool": identifying yourself by email gets better rate limits.
    # This previously sent the literal placeholder 'example@domain.com', so this app
    # never actually entered the polite pool. Set OPENALEX_MAILTO in .env; if it is
    # unset we simply omit the parameter rather than sending a fake address.
    if params is None:
        params = {}
    mailto = os.environ.get('OPENALEX_MAILTO')
    if mailto:
        params['mailto'] = mailto

    headers = {
        'User-Agent': 'OpenAlex_Scholarly_Dashboard/1.0',
        'Accept': 'application/json'
    }
    
    try:
        with st.spinner(f"Fetching data from OpenAlex {endpoint} endpoint..."):
            # Add delay between API calls to avoid rate limiting
            time.sleep(0.5)
            response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text[:100]}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to process data for the dashboard
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def process_data(topic, start_year, end_year, limit=200):
    """Process data for dashboard visualizations"""
    
    # NOTE: the OpenAlex parameter is `per-page` with a hyphen. This previously read
    # `per_page`, which the API ignores, so the page-size slider had no effect at all.
    #
    # `sort: publication_date:desc` was also removed. It returned only the newest N
    # papers, so every one fell in the same year and the "trends over time" chart had
    # a single bar. Without it OpenAlex sorts by relevance, giving a sample that
    # actually spans the requested range.
    params = {
        'filter': f'title.search:{topic},publication_year:{start_year}-{end_year}',
        'per-page': min(limit, 100),  # OpenAlex caps page size at 100
    }

    works_data = fetch_openalex_data('works', params)

    # If API call fails or no results, try different filter
    if not works_data or 'results' not in works_data or len(works_data['results']) == 0:
        # Try with abstract search instead
        params['filter'] = f'abstract.search:{topic},publication_year:{start_year}-{end_year}'
        works_data = fetch_openalex_data('works', params)

    # "No results" is a real, reportable outcome. Return None and let the caller render
    # an explicit empty state - never substitute invented numbers for missing data.
    if not works_data or 'results' not in works_data or len(works_data['results']) == 0:
        return None

    # Process the actual data
    works_df = pd.json_normalize(works_data['results'])
    
    # Extract basic information
    if 'publication_date' in works_df.columns:
        works_df['publication_year'] = pd.to_datetime(works_df['publication_date']).dt.year
    else:
        works_df['publication_year'] = works_df.get('publication_year', start_year)
    
    # Publication trends.
    #
    # Counted server-side with group_by rather than from the sampled page. group_by
    # aggregates over EVERY matching work, so these are true totals per year rather
    # than the shape of whichever 100 papers came back - a real trend line instead of
    # an artefact of the page size.
    trend_data = fetch_openalex_data('works', {
        'filter': f'title.search:{topic},publication_year:{start_year}-{end_year}',
        'group_by': 'publication_year',
    })
    groups = (trend_data or {}).get('group_by') or []
    if groups:
        rows = sorted(
            ((int(g['key']), g['count']) for g in groups if str(g.get('key','')).isdigit()),
            key=lambda kv: kv[0],
        )
        pub_trends_df = pd.DataFrame(rows, columns=['year', 'publications'])
    else:
        # Fall back to counting the sample, which is a weaker signal but still real.
        pub_trends = works_df['publication_year'].value_counts().sort_index()
        pub_trends_df = pd.DataFrame({'year': pub_trends.index, 'publications': pub_trends.values})
    
    # Citation metrics
    citation_data = works_df[['publication_year', 'cited_by_count']].copy() if 'cited_by_count' in works_df.columns else pd.DataFrame({
        'publication_year': works_df['publication_year'],
        'cited_by_count': 0
    })
    
    yearly_citations = citation_data.groupby('publication_year')['cited_by_count'].agg(['mean', 'sum', 'count'])
    yearly_citations.reset_index(inplace=True)
    yearly_citations.columns = ['year', 'avg_citations', 'total_citations', 'paper_count']
    
    # Open access trends. An absent column means the API did not return the field -
    # that is an empty result, not a reason to invent one.
    if 'open_access.is_oa' in works_df.columns:
        works_df['is_oa'] = works_df['open_access.is_oa'].fillna(False)
        oa_by_year = works_df.groupby('publication_year')['is_oa'].agg(['sum', 'count'])
        oa_by_year.columns = ['open_access', 'total']
        oa_by_year['percentage'] = (oa_by_year['open_access'] / oa_by_year['total'] * 100).round(1)
        oa_df = oa_by_year.reset_index().rename(columns={'publication_year': 'year'})
    else:
        oa_df = pd.DataFrame(columns=['year', 'open_access', 'total', 'percentage'])

    # Extract institution data.
    #
    # NOTE: this previously read `authorship['institution']` (singular). OpenAlex
    # exposes `institutions` as a LIST on each authorship, so that key was never
    # present, the list was always empty, and the chart silently fell back to a
    # hardcoded Harvard/Stanford/MIT table on every single run - including runs where
    # the API call had succeeded.
    inst_counts_map, inst_citations = {}, {}
    for work in works_data['results']:
        cited_by_count = work.get('cited_by_count', 0) or 0
        for authorship in work.get('authorships') or []:
            for inst in authorship.get('institutions') or []:
                name = inst.get('display_name')
                if not name:
                    continue
                inst_counts_map[name] = inst_counts_map.get(name, 0) + 1
                inst_citations.setdefault(name, []).append(cited_by_count)

    if inst_counts_map:
        top = sorted(inst_counts_map.items(), key=lambda kv: kv[1], reverse=True)[:10]
        inst_df = pd.DataFrame({
            'institution': [name for name, _ in top],
            'publications': [count for _, count in top],
            'total_citations': [sum(inst_citations[name]) for name, _ in top],
        })
    else:
        inst_df = pd.DataFrame(columns=['institution', 'publications', 'total_citations'])

    # Extract research topics/concepts if available
    concepts = [
        {'name': c.get('display_name', ''), 'score': c.get('score', 0)}
        for work in works_data['results']
        for c in (work.get('concepts') or [])
        if c.get('display_name')
    ]
    if concepts:
        concept_df = (
            pd.DataFrame(concepts)
            .groupby('name')['score'].mean().reset_index()
            .sort_values('score', ascending=False).head(15)
        )
    else:
        concept_df = pd.DataFrame(columns=['name', 'score'])

    # Summary statistics
    stats = {
        'total_papers': len(works_df),
        'total_citations': int(works_df['cited_by_count'].sum()) if 'cited_by_count' in works_df.columns else 0,
        'avg_citations': round(works_df['cited_by_count'].mean(), 1) if 'cited_by_count' in works_df.columns else 0,
        'h_index': calculate_h_index(works_df['cited_by_count'].tolist()) if 'cited_by_count' in works_df.columns else 0
    }
    
    return {
        'publication_trends': pub_trends_df,
        'citation_metrics': yearly_citations,
        'open_access': oa_df,
        'institutions': inst_df,
        'concepts': concept_df,
        'stats': stats
    }

def calculate_h_index(citations):
    """Calculate h-index from citation counts"""
    if not citations:
        return 0
    sorted_citations = sorted(citations, reverse=True)
    h = 0
    for i, citation in enumerate(sorted_citations, 1):
        if citation >= i:
            h = i
        else:
            break
    return h

def get_section(data_dict, key):
    """Return a section of the processed data, or None if it is absent or empty.

    Returns None rather than substituting invented data. Every caller is expected to
    render an explicit empty state so a reader can never mistake a gap in the data for
    a finding.
    """
    df = data_dict.get(key)
    if df is None or getattr(df, "empty", False):
        return None
    return df

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 OpenAlex Scholarly Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.title("Query Settings")
    
    # Input for research topic
    topic = st.sidebar.text_input("Research Topic", value="data visualization")
    
    # Date range
    current_year = datetime.now().year
    start_year = st.sidebar.number_input("Start Year", min_value=2010, max_value=current_year-1, value=current_year-5)
    end_year = st.sidebar.number_input("End Year", min_value=start_year+1, max_value=current_year, value=current_year)
    
    # Data limit
    data_limit = st.sidebar.slider("Number of Papers to Analyze", min_value=50, max_value=500, value=200, step=50)
    
    # Update button
    if st.sidebar.button("Update Dashboard"):
        st.session_state.update_clicked = True
        st.session_state.data = None  # Reset cached data
    
    # Initialize session state
    if 'update_clicked' not in st.session_state:
        st.session_state.update_clicked = False
    
    if 'data' not in st.session_state or st.session_state.data is None:
        if st.session_state.update_clicked:
            with st.spinner(f"Analyzing scholarly data on '{topic}' from {start_year} to {end_year}..."):
                st.session_state.data = process_data(topic, start_year, end_year, data_limit)
        else:
            # First time loading - use default data
            with st.spinner("Loading initial dashboard data..."):
                st.session_state.data = process_data("data visualization", current_year-5, current_year, 200)
    
    data = st.session_state.data

    # Explicit empty state. Previously this path returned fabricated trend lines that
    # rendered identically to real ones.
    if data is None:
        st.warning(
            f"OpenAlex returned no works for '{topic}' between {start_year} and "
            f"{end_year}. Nothing is shown below because there is nothing to show - "
            "try a broader topic or a wider year range."
        )
        return

    # Display summary statistics in cards
    st.markdown('<h2 class="section-header">Summary Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['total_papers']}</div>
            <div class="stat-label">Publications</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['total_citations']}</div>
            <div class="stat-label">Total Citations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['avg_citations']}</div>
            <div class="stat-label">Avg Citations per Paper</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['h_index']}</div>
            <div class="stat-label">h-index</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Publication and citation trends over time.
    #
    # process_data() has always computed 'publication_trends', 'citation_metrics' and
    # 'open_access', but main() never rendered any of them - three of six datasets were
    # calculated and thrown away, which left a "scholarly dashboard" with no time series
    # on it at all. Rendered below.
    st.markdown('<h2 class="section-header">Publication & Citation Trends</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        trends = get_section(data, 'publication_trends')
        if trends is None:
            st.info("No publication counts were returned for these works.")
        else:
            fig = px.bar(
                trends, x='year', y='publications',
                title='Publications per Year',
                labels={'year': 'Year', 'publications': 'Publications'},
                template='plotly_white',
            )
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Counted across every matching work in OpenAlex, not the sample below.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cites = get_section(data, 'citation_metrics')
        if cites is None:
            st.info("No citation counts were returned for these works.")
        else:
            fig = px.line(
                cites, x='year', y='avg_citations', markers=True,
                title='Average Citations per Paper by Year',
                labels={'year': 'Year', 'avg_citations': 'Avg. citations'},
                template='plotly_white',
            )
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"From the {data['stats']['total_papers']} papers analysed. Recent years "
                "trend lower because citations accrue over time, not because the work is "
                "less cited."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Open access share
    st.markdown('<h2 class="section-header">Open Access Share</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    oa = get_section(data, 'open_access')
    if oa is None:
        st.info("OpenAlex returned no open-access information for these works.")
    else:
        fig_oa = px.bar(
            oa, x='year', y='percentage',
            title='Share of Publications that are Open Access (%)',
            labels={'year': 'Year', 'percentage': '% open access'},
            range_y=[0, 100], template='plotly_white',
        )
        fig_oa.update_xaxes(dtick=1)
        st.plotly_chart(fig_oa, use_container_width=True)
        st.caption(f"From the {data['stats']['total_papers']} papers analysed, not all of OpenAlex.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Institutions and Research Topics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h2 class="section-header">Top Contributing Institutions</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        inst_data = get_section(data, 'institutions')
        if inst_data is None:
            st.info("No institution affiliations were returned for these works.")
        else:
            fig_inst = px.scatter(
                inst_data,
                x='publications',
                y='total_citations',
                size='publications',
                color='total_citations',
                hover_name='institution',
                text='institution',
                title='Institutions by Publications and Citations',
                labels={'publications': 'Number of Publications', 'total_citations': 'Total Citations'},
                template='plotly_white'
            )
            
            fig_inst.update_traces(
                textposition='top center',
                marker=dict(sizemode='area', sizeref=0.1)
            )
            
            st.plotly_chart(fig_inst, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Related Research Topics</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        concept_data = get_section(data, 'concepts')
        if concept_data is None:
            st.info("No concept tags were returned for these works.")
        else:
            fig_concept = px.bar(
                concept_data,
                x='score',
                y='name',
                orientation='h',
                title='Top Related Research Concepts',
                labels={'score': 'Relevance Score', 'name': 'Concept'},
                color='score',
                color_continuous_scale=px.colors.sequential.Viridis,
                template='plotly_white'
            )
            
            fig_concept.update_layout(
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_concept, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #f0f2f6;">
        <p>Powered by OpenAlex API • Data retrieved on {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()