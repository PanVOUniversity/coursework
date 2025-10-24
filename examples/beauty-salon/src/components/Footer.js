import React from 'react';
import { motion } from 'framer-motion';
import { FaCut, FaPhone, FaEnvelope, FaInstagram, FaFacebook } from 'react-icons/fa';
import './styles.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <motion.div className="footer-section" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            <div className="footer-logo">
              <FaCut />
              <span>Beauty Salon</span>
            </div>
            <p>Ваша красота - наша работа</p>
            <div className="social-links">
              <motion.a href="#" whileHover={{ scale: 1.1 }}><FaInstagram /></motion.a>
              <motion.a href="#" whileHover={{ scale: 1.1 }}><FaFacebook /></motion.a>
            </div>
          </motion.div>

          <motion.div className="footer-section" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            <h3>Контакты</h3>
            <p><FaPhone /> +7 (495) 123-45-67</p>
            <p><FaEnvelope /> info@beautysalon.ru</p>
          </motion.div>
        </div>

        <div className="footer-bottom">
          <p>&copy; 2024 Beauty Salon. Все права защищены.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
