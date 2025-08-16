function getRecommendations(input, callback) {
  fetch('https://shriyawealth.onrender.com/recommend', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(input)
  })
  .then(response => response.json())
  .then(data => callback(data.recommendations))
  .catch(err => {
    document.getElementById("advisorResults").innerHTML = `<div style="color:red;">API Error: ${err}</div>`;
  });
}

document.addEventListener("DOMContentLoaded", function() {
  document.getElementById("advisorForm").onsubmit = function(e) {
    e.preventDefault();
    const form = e.target;
    const input = Object.fromEntries(new FormData(form).entries());
    input.amount = parseInt(input.amount);
    input.tenor = parseInt(input.tenor);
    getRecommendations(input, function(funds) {
      const resDiv = document.getElementById("advisorResults");
      if (!funds || funds.length === 0) {
        resDiv.innerHTML = "<p class='no-results'>No suitable funds found under current criteria.</p>";
        return;
      }
      resDiv.innerHTML = "<h2>Recommended Funds:</h2>" +
        "<ol>" + funds.map(f =>
          `<li>
            <b>${f.name}</b> (${f.type}) <br>
            <span class="xirr">XIRR: ${f.xirr}%, Bear: ${f.bear}%, Bull: ${f.bull}%</span><br>
            <span class="explanation">${f.explanation}</span>
          </li>`
        ).join("") + "</ol>" +
        "<p class='disclaimer'>This is an AI-powered suggestion based on your input. Please consult a SEBI-registered advisor before investing.</p>";
    });
  };
});
