---
layout: page
title: Notebook
description: Notes of Lester James V. Miranda
permalink: /notebook/
---

Notebook documentation  

<ul>
  {% for post in site.categories.notebook %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
