import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =========================
# SAFE GAT IMPORTS
# =========================
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
    GAT_AVAILABLE = True
except:
    GAT_AVAILABLE = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Satellite Collision Intelligence System", layout="wide")

# =========================
# HIGH VISIBILITY UI STYLING
# =========================
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0a0f1c, #1a2a44, #2c5364);
        color: #e0f0ff;
    }

    h1 {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(to right, #00f5ff, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 198, 255, 0.5);
    }

    .risk-card {
        border-radius: 18px;
        padding: 32px 20px;
        margin: 18px 0;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 12px 50px rgba(0, 0, 0, 0.8);
        border: 3px solid;
    }

    .very-dangerous {
        background: #b3002d;
        border-color: #ff3366;
        color: white;
        text-shadow: 0 0 15px #ff3366;
    }

    .medium-risk {
        background: #cc7700;
        border-color: #ffcc00;
        color: white;
    }

    .low-risk {
        background: #006633;
        border-color: #00ff99;
        color: white;
    }

    .confidence-card {
        background: #003366;
        border: 4px solid #00ccff;
        border-radius: 18px;
        padding: 25px 20px;
        margin: 18px 0;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        color: white;
        box-shadow: 0 0 25px #00ccff;
        text-shadow: 0 0 12px #00f5ff;
    }

    .blink {
        animation: blink 1.3s infinite;
    }

    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA + FUNCTIONS
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("/Users/ats/Desktop/ProjectM/data/space_objects.csv")

data = load_data()

def compute_semi_major_axis(mean_motion):
    mu = 398600
    n = mean_motion * 2 * np.pi / 86400
    return (mu / (n ** 2)) ** (1/3)

def generate_orbit_points(a, e, i, raan, argp, M0, num_points=300):
    t = np.linspace(0, 2*np.pi, num_points)
    r = a * (1 - e**2) / (1 + e * np.cos(t))
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = r * np.sin(np.radians(i))
    return np.vstack((x, y, z)).T

def compute_min_distance(p1, p2):
    return np.min(np.linalg.norm(p1 - p2, axis=1))

# =========================
# LOAD GAT MODEL
# =========================
if GAT_AVAILABLE:
    class GATModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gat1 = GATConv(6, 32)
            self.gat2 = GATConv(32, 16)
            self.fc = torch.nn.Linear(16, 3)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.gat1(x, edge_index)
            x = F.relu(x)
            x = self.gat2(x, edge_index)
            x = F.relu(x)
            return self.fc(x)

    model = GATModel()
    try:
        model.load_state_dict(torch.load("gat_modelll.pth", map_location="cpu"))
        model.eval()
    except:
        model = None
else:
    model = None

# =========================
# HEADER + SIDEBAR
# =========================
st.title("🛰️ Satellite Collision Intelligence System")
st.markdown("<p style='text-align:center; color:#88ccff; font-size:18px;'>Orbital Risk Assessment</p>", unsafe_allow_html=True)

st.sidebar.header("📡 Satellite Input Parameters")

a = st.sidebar.number_input("Semi-major Axis (km)", value=7000.0, step=10.0)
e = st.sidebar.number_input("Eccentricity", value=0.001, format="%.5f", step=0.0001)
i = st.sidebar.number_input("Inclination (deg)", value=98.0, step=0.1)
raan = st.sidebar.number_input("RAAN (deg)", value=0.0, step=0.1)
argp = st.sidebar.number_input("Argument of Perigee (deg)", value=0.0, step=0.1)
M0 = st.sidebar.number_input("Mean Anomaly (deg)", value=0.0, step=0.1)

run = st.sidebar.button("🚀 Run Simulation", type="primary")

# =========================
# MAIN
# =========================
if run:
    with st.spinner("Calculating orbits and risk..."):
        user_orbit = generate_orbit_points(a, e, i, raan, argp, M0)

        results = []
        for _, row in data.iterrows():
            try:
                a2 = compute_semi_major_axis(row['MEAN_MOTION'])
                orbit = generate_orbit_points(
                    a2, row['ECCENTRICITY'], row['INCLINATION'],
                    row['RA_OF_ASC_NODE'], row['ARG_OF_PERICENTER'], row['MEAN_ANOMALY']
                )
                dist = compute_min_distance(user_orbit, orbit)
                results.append((row['OBJECT_NAME'], dist, orbit))
            except:
                continue

        results_sorted = sorted(results, key=lambda x: x[1])[:5]
        min_distance = results_sorted[0][1]

        # ===================== TABS =====================
        tab1, tab2, tab3 = st.tabs(["📊 Risk Overview", "🌍 3D Orbit View", "📋 Detailed Results"])

        # TAB 1: Risk Overview
        with tab1:
            st.subheader("Risk Assessment")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Closest Approach", f"{min_distance:.2f} km")
            with col_m2:
                st.metric("Objects Analyzed", len(data))

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("🚨 Top 5 Closest Satellites")
                for name, dist, _ in results_sorted:
                    st.markdown(f"**{name}** → **{dist:.2f} km**")

            with col2:
                if min_distance < 25:
                    st.markdown('<div class="risk-card very-dangerous blink">🚨 VERY DANGEROUS</div>', unsafe_allow_html=True)
                elif min_distance <= 75:
                    st.markdown('<div class="risk-card medium-risk">⚠️ MEDIUM RISK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-card low-risk">✅ LOW RISK</div>', unsafe_allow_html=True)

            # Graph Attention Network Prediction
            st.subheader("🤖 Graph Attention Network (GAT) Prediction")
            if model is not None:
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.neighbors import NearestNeighbors

                    features = [[compute_semi_major_axis(row['MEAN_MOTION']), row['ECCENTRICITY'], row['INCLINATION'],
                                 row['RA_OF_ASC_NODE'], row['ARG_OF_PERICENTER'], row['MEAN_ANOMALY']] 
                                for _, row in data.iterrows()]

                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    x = torch.tensor(features_scaled, dtype=torch.float)

                    k = 5
                    nbrs = NearestNeighbors(n_neighbors=k).fit(features_scaled)
                    _, indices = nbrs.kneighbors(features_scaled)

                    edges = [[i, indices[i][j]] for i in range(len(features)) for j in range(1, k)]

                    user_feat = np.array([[a, e, i, raan, argp, M0]])
                    user_feat_scaled = scaler.transform(user_feat)
                    user_tensor = torch.tensor(user_feat_scaled, dtype=torch.float)

                    new_x = torch.cat([x, user_tensor], dim=0)
                    user_idx = len(new_x) - 1

                    for idx in range(len(x)):
                        edges.append([idx, user_idx])
                        edges.append([user_idx, idx])

                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    graph = Data(x=new_x, edge_index=edge_index)

                    with torch.no_grad():
                        output = model(graph)

                    logits = output[-1]
                    probs = torch.softmax(logits, dim=0)
                    conf = max(probs).item()

                    st.markdown(f"""
                    <div class="confidence-card">
                        🤖 Graph Attention Network Confidence<br>
                        <span style="font-size:32px; color:#00ffff;">{conf:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if min_distance > 100:
                        st.markdown('<div class="risk-card low-risk">✅ LOW RISK</div>', unsafe_allow_html=True)
                    elif probs[2].item() > 0.6:
                        st.markdown('<div class="risk-card very-dangerous blink">🚨 HIGH RISK</div>', unsafe_allow_html=True)
                    elif probs[1].item() > 0.5:
                        st.markdown('<div class="risk-card medium-risk">⚠️ MEDIUM RISK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-card low-risk">✅ LOW RISK</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.warning("GAT prediction unavailable")
            else:
                st.warning("GAT model not loaded")

        # ===================== TAB 2: 3D VIEW - ALL 5 ORBITS (Only User Marker) =====================
        with tab2:
            st.subheader("🌍 3D Orbit Visualization")
            st.caption("Cyan = Your Satellite | Red = Closest | Orange = Medium Risk | Green = Low Risk")

            fig = go.Figure()

            # Earth
            R = 6371
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi, 60)
            fig.add_trace(go.Surface(
                x=R*np.outer(np.cos(u), np.sin(v)),
                y=R*np.outer(np.sin(u), np.sin(v)),
                z=R*np.outer(np.ones_like(u), np.cos(v)),
                opacity=0.65, colorscale='Blues', showscale=False
            ))

            # User Orbit + Marker (Only user has marker)
            fig.add_trace(go.Scatter3d(
                x=user_orbit[:,0], y=user_orbit[:,1], z=user_orbit[:,2],
                mode='lines', line=dict(color='cyan', width=6), name="Your Orbit"
            ))
            fig.add_trace(go.Scatter3d(
                x=[user_orbit[0,0]], y=[user_orbit[0,1]], z=[user_orbit[0,2]],
                mode='markers+text',
                marker=dict(size=16, color='cyan', line=dict(width=4, color='white')),
                text=" YOUR SATELLITE",
                textposition="top center",
                name="Your Satellite"
            ))

            # All Top 5 Closest Satellites Orbits (No markers)
            for idx, (name, dist, orbit) in enumerate(results_sorted):
                if idx == 0:           # Closest
                    color = 'red'
                    width = 5
                elif dist <= 75:       # Medium risk
                    color = 'orange'
                    width = 4
                else:                  # Low risk
                    color = 'lime'
                    width = 4

                fig.add_trace(go.Scatter3d(
                    x=orbit[:,0], y=orbit[:,1], z=orbit[:,2],
                    mode='lines',
                    line=dict(color=color, width=width),
                    name=name
                ))

            # Layout
            fig.update_layout(
                scene=dict(
                    bgcolor="#050a14",
                    xaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
                    zaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.4))
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=780,
                legend=dict(bgcolor="rgba(10,15,40,0.95)", font=dict(color="white", size=12))
            )

            st.plotly_chart(fig, use_container_width=True)

        # ===================== TAB 3: DETAILED RESULTS =====================
        with tab3:
            st.subheader("📋 Detailed Results")
            results_df = pd.DataFrame(
                [(name, round(dist, 2)) for name, dist, _ in results_sorted],
                columns=["Object Name", "Minimum Distance (km)"]
            )
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="satellite_collision_risk_results.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Enter your satellite parameters in the sidebar and click **Run Simulation**")