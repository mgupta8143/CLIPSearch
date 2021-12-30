import type { NextPage } from 'next'
import styles from '../styles/Home.module.css'
import Search  from '../templates/Search';

const Home: NextPage = () => {

  return (
    <div className={styles.flexBody}>
      <div className={styles.leftCol}>
        <Search/>
      </div>
      <div className={styles.rightCol}>
        Hello
      </div>
    </div>
  )
}

export default Home
