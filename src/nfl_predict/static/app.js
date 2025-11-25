document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  const results = document.getElementById('results');
  const tableBody = document.querySelector('#pred-table tbody');
  const errorBox = document.getElementById('error');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    results.classList.add('hidden');
    errorBox.classList.add('hidden');
    tableBody.innerHTML = '';

    const season = document.getElementById('season').value || null;
    const week = document.getElementById('week').value || null;
    const position = document.getElementById('position').value;
    const top_n = parseInt(document.getElementById('top_n').value, 10) || 25;

    const payload = { season: season ? Number(season) : null, week: week ? Number(week) : null, position, top_n };

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`Server error: ${resp.status} ${txt}`);
      }

      const data = await resp.json();
      if (!data || !data.predictions) {
        throw new Error('Unexpected response from server');
      }

      data.predictions.forEach((p) => {
        const tr = document.createElement('tr');
        const nameTd = document.createElement('td');
        nameTd.textContent = p.player_name || 'Unknown';
        const teamTd = document.createElement('td');
        teamTd.textContent = p.team || '';
        const ptsTd = document.createElement('td');
        ptsTd.textContent = (Math.round((p.expected_ppr_points || 0) * 1000) / 1000).toFixed(3);
        tr.appendChild(nameTd);
        tr.appendChild(teamTd);
        tr.appendChild(ptsTd);
        tableBody.appendChild(tr);
      });

      results.classList.remove('hidden');
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.classList.remove('hidden');
    }
  });
});
