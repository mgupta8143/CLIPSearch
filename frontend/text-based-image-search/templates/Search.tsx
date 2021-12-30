import type { NextPage } from 'next'
import { useState } from 'react';
import styles from '../styles/Search.module.css'

const axios = require('axios');

const Search: NextPage = (props) => {
    let [searchText, setSearchText] = useState("");

    const handleSearchChange = (event: any) => {
        setSearchText(event.target.value);
    }

    const handleSubmit = async (event: any) => {
        event.preventDefault();
        console.log(searchText);

        const API_URL = "http://34.226.119.97/get_images";
        const NUM_IMAGES = 10;


        const response = await axios.post(API_URL, {
            "search_query": searchText,
            "num_images": NUM_IMAGES
        });

        console.log(response);


    }

    return (
      <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={searchText}
            onChange={handleSearchChange}
          />
      </form>
    )
  }
  
  export default Search
  