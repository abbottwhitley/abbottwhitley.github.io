---
layout: page
title: Documentation
description: Projects documentation for Julian Abbott
permalink: /documentation/
---

Project documentation  

<ul>
  {% for post in site.categories.documentation %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
