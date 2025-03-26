---
layout: default
---

{% for post in site.posts %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
  <small>{{ post.date | date: "%B %d, %Y" }}</small>
  <p>{{ post.description }}</p>
{% endfor %}

